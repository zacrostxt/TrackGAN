import tensorflow as tf
import numpy as np
# Tensorflow Dist
import tensorflow_probability as tfp
# Custom lib
import image_utils

class GAN_wrapper(tf.keras.Model):
  
    def __init__(self, discriminator, generator, batch_size, noise_dim, num_classes, seed = 1234, gan_metrics = []):
      
      super(GAN_wrapper, self).__init__()

      # Architecture
      self.discriminator = discriminator
      self.generator = generator

      self.noise_dim = noise_dim
      self.batch_size = batch_size
      self.num_classes = num_classes
      # Gan Metrics
      self.gan_metrics = gan_metrics
      #Random seed
      self.seed = seed

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss):
      #super.compile()
      super(GAN_wrapper, self).compile()
      # Losses
      self.discriminator_loss = discriminator_loss
      self.generator_loss = generator_loss
      
      # Optimizers
      self.discriminator_optimizer = discriminator_optimizer
      self.generator_optimizer = generator_optimizer
      
    


class InfoGAN_wrapper(GAN_wrapper):
    def __init__(self, discriminator, generator, q_model, batch_size, noise_dim, num_classes, seed = 1234, gan_metrics = []):
        # Old Lib?
        super().__init__(discriminator, generator, batch_size, noise_dim, num_classes, seed = 1234, gan_metrics = gan_metrics)
        # Architecture
        self.q_model = q_model



    def compile(self, discriminator_optimizer, generator_optimizer, q_optimizer, discriminator_loss, generator_loss):
      #super.compile()
      super().compile(discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss)
      # Losses
      self.q_cat_loss = tf.keras.losses.CategoricalCrossentropy() 
      # Optimizers
      self.q_optimizer = q_optimizer



    def concat_inputs(self, input):
        concat_input = tf.keras.layers.Concatenate()(input)
        return concat_input
    

    # Sample and display some input
    def sample_from_model(self, num_examples = 1):
      
      print(" Variations of the continous variable (Column) ")

      for i in range(0,5):
        
        # Sample some input
        label, c_1, noise = self.sample_noise(batch_size = 1)

        # Replicate the same input several times
        label = np.array( [ label[0]]*num_examples )
        c_1 = np.array( [ c_1[0]]*num_examples )
        noise = np.array( [ noise[0]]*num_examples )

        # Explore the continous value conditioning
        c_1 = tf.sort( tf.random.uniform( [num_examples,1] , minval=-5, maxval=3), axis=0 )
       

        # Batch data
        gen_input = self.concat_inputs( [label, c_1, noise] )
        # Generate Images
        generated_images = self.generator(gen_input, training= False)
        # Plot the Images
        image_utils.display_multiple_image(image_utils.denormalize_images(generated_images) ,  size = (900,900), inline=True)

    

    def sample_noise(self, batch_size):
      # create noise input
      noise = tf.random.normal([batch_size, self.noise_dim], seed=self.seed)
      # Create categorical latent code
      label = tf.random.uniform([batch_size], minval=0, maxval=self.num_classes, dtype=tf.int32, seed=self.seed)
      label = tf.one_hot(label, depth=self.num_classes)
      # Create one continuous latent code
      c_1 = tf.random.uniform([batch_size, 1], minval=-1, maxval=1, seed=self.seed)

      
      return label, c_1, noise

    
    # Perform a training step over the input
    def train_step(self, disc_input):


      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as q_tape:

          # Train the Discriminator K times - In here we have access on only 1 batch. Need to relocate if needed
          for k in range(0,1):

            #Allow the discriminator to be trained
            self.discriminator.trainable= True

            # Sample Noise
            g_label, c_1, gen_noise = self.sample_noise(self.batch_size)


            gen_input = self.concat_inputs( (g_label, c_1, gen_noise) )
          
            # Generate Images given inputs
            generated_images = self.generator( inputs = gen_input , training=True)


            # Output of the discriminator based on fake images
            fake_output = self.discriminator( generated_images , training=True)
            # Output of the discriminator based on real images
            real_output = self.discriminator( disc_input, training=True)

            # Discriminator loss based on real and fake images
            disc_loss , real_loss , fake_loss = self.discriminator_loss(real_output, fake_output)


            # Discriminator Gradient
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Discriminator Backprop
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
          

          # Train the Generator
      
    
          # Sample Noise
          g_label, c_1, gen_noise = self.sample_noise(self.batch_size)

          gen_input = self.concat_inputs( [g_label, c_1, gen_noise] )

          # Generate Images given Noise
          generated_images = self.generator( gen_input , training=True)

          # Output of the discriminator based on fake images
          disc_output = self.discriminator( generated_images , training=True)

          # Generator loss based only on fake images
          gen_loss = self.generator_loss(disc_output)


          # AUX model loss
          

          cat_output, mu, sigma = self.q_model(generated_images, training=True)
          # Categorical loss
          cat_loss = self.q_cat_loss(g_label, cat_output)
          # Use Gaussian distributions to represent the output
          dist = tfp.distributions.Normal(loc=mu, scale=sigma)
          # Losses (negative log probability density function as we want to maximize the probability density function)
          c_1_loss = tf.reduce_mean(-dist.log_prob(c_1))
          
          # Auxiliary function loss
          q_loss = (cat_loss + 0.1*c_1_loss)
          #g_loss = g_img_loss + q_loss
          
          # Do not update the discriminator part, just the q network
          self.discriminator.trainable= False

          q_gradients = q_tape.gradient(q_loss, self.q_model.trainable_variables)
          # Optimize
          self.q_optimizer.apply_gradients(zip(q_gradients, self.q_model.trainable_variables))
              


          
          # Update Generator loss
          gen_loss = gen_loss + q_loss
          
          # Generator Gradient
          gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
          # Generator Backprop
          self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))



          # Compute some metrics 
          for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
              gan_metric.update_metric(output_on_real_images=real_output, output_on_fake_images=fake_output )

            if gan_metric.GAN_module == 'generator':
              gan_metric.update_metric( disc_output= disc_output)
            
            if gan_metric.GAN_module == 'single_input_metric':
              gan_metric.update_metric( q_loss )




      #return tf.math.reduce_mean(gen_loss) , tf.math.reduce_mean(disc_loss) , tf.math.reduce_mean(real_loss) , tf.math.reduce_mean(fake_loss)




class CGAN_wrapper(GAN_wrapper):
    def __init__(self, discriminator, generator, batch_size, noise_dim, num_classes, seed = 1234, gan_metrics = []):
        # Old Lib?
        super().__init__(discriminator, generator, batch_size, noise_dim, num_classes, seed = 1234, gan_metrics = gan_metrics)
        


    def compile(self, discriminator_optimizer, generator_optimizer , discriminator_loss, generator_loss):
      #super.compile()
      #super().compile(discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss)
      super().compile(discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss)
    
    def sample_noise(self):
      return tf.random.normal( shape = [self.batch_size, self.noise_dim] )   
    

    def sample_from_model(self):
      num_examples_to_generate = 10

      noise_sample = train_asses.generate_noise([ num_examples_to_generate, noise_dim])


      for i in range(0,4):
        #Set the target index to 1
        sample_target = [0,0,0,0,0,0,0,0,0,0]
        sample_target[i] = 1
        print("Target : " , i)

        # Batch data
        batch_target = np.array([ sample_target ]*num_examples_to_generate)

        generated_images = self.generator([noise_sample , batch_target],training= False)
        image_utils.display_multiple_image(image_utils.denormalize_images(generated_images) ,  size = (900,900), inline=True)
    


    def train_step(self, disc_input, gen_extra_inputs = None):

      # gen_extra_inputs - a 'flexible' way of adding additional inputs to the generator
      # TRYING A RANDOM GENERATION OF LABELS !!!!
      #labels_gen = randint(0, 10, BATCH_SIZE)
      #labels_gen = tf.one_hot(labels_gen, 10)
      #extra_inputs = labels_gen



      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Train the Discriminator K times - In here we have access on only 1 batch. Need to relocate if needed
        for k in range(0,1):

          # Sample Noise
          noise = self.sample_noise()
          #print("NOISE SHAPE : ",gen_input.shape )
          #print("LABELS SHAPE : ",gen_extra_inputs.shape )

          # Just to have explicit names and clarity
          gen_input = noise
          # Check for additional extra inputs for the generator outside the noise
          if gen_extra_inputs is not None:
            gen_input =[gen_input , gen_extra_inputs ]
    
          # Generate Images given inputs
          generated_images = self.generator( inputs = gen_input , training=True)

          # Add additional provided inputs
          disc_fake_input = generated_images
          if gen_extra_inputs is not None:
            disc_fake_input =[disc_fake_input , gen_extra_inputs ]


          # Output of the discriminator based on fake images
          fake_output = self.discriminator( disc_fake_input , training=True)
          # Output of the discriminator based on real images
          real_output = self.discriminator( disc_input, training=True)

          # Discriminator loss based on real and fake images
          disc_loss , real_loss , fake_loss = self.discriminator_loss(real_output, fake_output)


          # Discriminator Gradient
          gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
          # Discriminator Backprop
          self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        

        # Train the Generator
    
        # Sample Noise
        gen_input = self.sample_noise()
        
        # Check for additional extra inputs for the generator outside the noise
        if gen_extra_inputs is not None:
          gen_input = [ gen_input , gen_extra_inputs ]
        # Generate Images given Noise
        generated_images = generator( gen_input , training=True)

        # Add additional provided inputs
        disc_fake_input = generated_images
        if gen_extra_inputs is not None:
          disc_fake_input =[disc_fake_input , gen_extra_inputs ]

        # Output of the discriminator based on fake images
        disc_output = self.discriminator( disc_fake_input , training=True)

        # Generator loss based only on fake images
        gen_loss = self.generator_loss(disc_output)
        
        
        
        # Generator Gradient
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # Generator Backprop
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))



        # Compute some metrics 
        for gan_metric in self.gan_metrics:
          # Check if it is a metric for the discriminator
          if gan_metric.GAN_module == 'discriminator':
            gan_metric.update_metric(output_on_real_images=real_output, output_on_fake_images=fake_output )

          if gan_metric.GAN_module == 'generator':
            gan_metric.update_metric( disc_output= disc_output)
        
        
        

      #return tf.math.reduce_mean(gen_loss) , tf.math.reduce_mean(disc_loss) , tf.math.reduce_mean(real_loss) , tf.math.reduce_mean(fake_loss)

  


