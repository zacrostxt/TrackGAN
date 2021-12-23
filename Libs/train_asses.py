import tensorflow as tf
import numpy as np
# Tensorflow Dist
import tensorflow_probability as tfp
tfd = tfp.distributions
from collections import deque # Used to have limited arrays

import re #regex

import image_utils

import time
# Manipulate the cells output
from IPython import display








#IN case of switch into iterations

#batches_per_epoch = floor(dataset_size / batch_size)
#total_iterations = batches_per_epoch * total_epochs

def train(dataset, epochs , generator, discriminator ,generator_optimizer , discriminator_optimizer ,generator_loss , discriminator_loss ,
          summary_writer,checkpoint_manager ,gen_seed,distributions = None ,valid_dataset = None,  starting_epoch = 0 , model_name = 'default_name',
          noise_dim = 100 , image_save_path = None, DEBUG = False, BATCH_SIZE = 128):
  

  total_batches = dataset.cardinality().numpy()
  # Used to add some noise to zero probabilities - HARD CODED BAD
  epsilon = 1e-3

  # Keep last 5 generated image to compare
  image_collection_deque = deque( maxlen= 5)

  # [Real Accuracy, Fake Accuracy]
  discriminator_accuracy = np.empty( (0,2), float)

  for epoch in range(starting_epoch , epochs):
    start = time.time()
    gen_loss_per_batch = []
    disc_loss_per_batch = []

    # Discriminator Decomposite Loss
    disc_real_loss_per_batch = []
    disc_fake_loss_per_batch = []

    
    
    step = 0

    # Training Step
    for inputs in dataset:
      # Assuming there's no extra inputs to pass to the generator
      extra_inputs = None
      # This means that there's multiple inputs, hoping tha n_inputs != batch_size :D
      if len(inputs) != BATCH_SIZE: # NEED TO CHANGE OR BATCH WITH DIFFERENT SIZE (DROP REMANINDER FALSE ) GOES ERROr
        # Extracting the inputs for the generator
        extra_inputs = inputs[1:]

      #print(image_batch.shape)
      #print(labels_batch.shape)

      #gen_loss , disc_loss = train_step(image_batch, labels_batch )
      #gen_loss , disc_loss , real_loss , fake_loss = improved_train_step(image_batch, labels_batch )
      gen_loss , disc_loss , real_loss , fake_loss = train_step( inputs , noise_dim , generator, discriminator ,
                                                                                    generator_optimizer , discriminator_optimizer ,
                                                                                    generator_loss , discriminator_loss , gen_extra_inputs = extra_inputs , BATCH_SIZE = BATCH_SIZE )

      # Store History
      gen_loss_per_batch.append(gen_loss)
      disc_loss_per_batch.append(disc_loss)
      disc_real_loss_per_batch.append(real_loss)
      disc_fake_loss_per_batch.append(fake_loss)
    
      
      # Every x do something
      do_every = 60

      # Training step
      if (step+1) % do_every == 0:
        #display.clear_output(wait=True)

        n_bars = int(((step+1)/do_every))
        print(f"Epoch {epoch} : [{step}:{total_batches}] <" + '-'*n_bars + ' '*(int(total_batches/do_every) - n_bars) + ">")

    
      step +=1
    

      # DEBUGGING
      if DEBUG:
        if (step+1) % 3 == 0:
          break
    
    
    # Average the losses per batch
    avg_gen_loss = tf.math.reduce_mean(gen_loss_per_batch)
    avg_disc_loss = tf.math.reduce_mean(disc_loss_per_batch)
    avg_disc_real_loss_per_batch = tf.math.reduce_mean(disc_real_loss_per_batch)
    avg_disc_fake_loss_per_batch = tf.math.reduce_mean(disc_fake_loss_per_batch)

    # Reset Values
    gen_loss_per_batch = []
    disc_loss_per_batch = []
    disc_real_loss_per_batch = []
    disc_fake_loss_per_batch = []







    # Validation Step
    if valid_dataset:
      disc_acc_on_real_image , disc_acc_on_fake_image = valid_step(valid_dataset)
      discriminator_accuracy = np.append( discriminator_accuracy , [ [disc_acc_on_real_image,disc_acc_on_fake_image] ] , axis = 0 )
      #discriminator_accuracy.append( [disc_acc_on_real_image , disc_acc_on_fake_image] )

        
    
        
    
    
    # Save the model and do stuff every x epochs
    if (epoch + 1) % 1 == 0:
      display.clear_output(wait=True)
      
      save_path = checkpoint_manager.save()
      #save_path = checkpoint.save(file_prefix = 'ckpt')
      print("Saved checkpoint for step {}: {}\n".format(int(checkpoint_manager.checkpoint.step), save_path))
      print("Generator Epoch loss {:1.3f}".format(avg_gen_loss.numpy())  )
      print("Discriminator Epoch Total Loss {:1.3f}  Discriminator Epoch Real Loss {:1.3f}  Discriminator Epoch Fake Loss {:1.3f}\n".format(avg_disc_loss.numpy() ,avg_disc_real_loss_per_batch.numpy() , avg_disc_fake_loss_per_batch.numpy()) )

      
      
      # Validation Score So Far
      #print("Accuracy on Real Images Last 10 Epoch : {}".format( np.around(discriminator_accuracy[-10:,0],3)) )
      #print("Accuracy on Fake Images Last 10 Epoch : {}".format( np.around(discriminator_accuracy[-10:,1],3)) )

      

    # Produce images for the GIF as you go   
      # Save Image every x epochs - Due to Memory Constraint
    save_image_condition = (epoch + 1) % 3 == 0
    image_name = 'image_at_epoch_{:04d}.png'.format(epoch+1)
    save_param = {"Save Image" : save_image_condition, "Path" : image_save_path, "filename" : image_name}
    
    # Generate Images From The Seed
    generated_seed_images = generator(gen_seed, training=False)
    denorm_generated_seed_images = image_utils.denormalize_images(generated_seed_images)


    '''
    # COMPUTE LOG PROB on normalized images (the dataset was normalized when the mean and std got extracted)
    mean_per_px_gen , _ = image_utils.get_mean_std_per_pixel(generated_seed_images + epsilon)
    mean_per_channel_gen , std_per_channel_gen = image_utils.get_mean_std_per_channel(generated_seed_images + epsilon)
    # Per Patch
    mean_per_patch_channel , _ = image_utils.get_mean_std_per_patch( generated_seed_images + epsilon , patch_shape  = [2,2] , patch_type = 'channel' )


    results = {}
    for key,dist in destributions:
      results[key] = 


    px_log_prob = get_image_sum_log_prob(mean_per_px_gen , px_dist)
    ch_log_prob = get_image_sum_log_prob(mean_per_channel_gen , ch_px_dist)
    # Per Patch Loss
    patch_ch_log_prob = get_image_sum_log_prob(mean_per_patch_channel , patch_ch_px_dist)
    

    print("\nPer Pixel Log Prob   {:.3f} \nPer Channel Log Prob   {:.3f} ".format(px_log_prob , ch_log_prob))
    print("Per (2,2 )Patch Channel Log Prob   {:.3f} \n".format(patch_ch_log_prob))

    # Log Accuracy on TB
    with summary_writer.as_default(step=epoch):
      tf.summary.scalar('Px Log Prob', px_log_prob)
      tf.summary.scalar('Channel Log Prob', ch_log_prob)
      tf.summary.scalar('(2,2) Patch Channel Log Prob ', patch_ch_log_prob)
    '''
    if distributions is not None:
      avg_log_loss_res = extract_distribution_log_losses(distributions , denorm_generated_seed_images)



    # Display and Save the Generated Image
    image_utils.display_multiple_image(denorm_generated_seed_images , size = (460,460) , inline=False , save_param = save_param , denormalize = False )

    
    # Stack multiple images generated into a single one
    stacked_image = image_utils.stack_multiple_images(denorm_generated_seed_images)
    #stacked_image = image_utils.denormalize_images(stacked_image)
    #image_utils.display_image(stacked_image ,  modify_size_by = 2)

    # Keep Track Of latest Generations
    image_collection_deque.append(stacked_image)
        

    


    # Display latest generated images
    #for j in range( len(image_collection_deque)-1 ): # Don't show the newly added image
    #for image_collection in image_collection_deque:
    # Display
    image_utils.display_multiple_image(image_collection_deque , size = (600,600) , inline=True  )

    # TO ADJUST
    
    image_utils.plot_channels_dist(distributions['Distribution Per Channel'] , title = "Original Data Channel Color Distributions")
    mean_per_channel_gen , std_per_channel_gen = image_utils.get_mean_std_per_channel(denorm_generated_seed_images)
    image_utils.plot_channels_dist(tfd.Normal(loc=mean_per_channel_gen+epsilon, scale=std_per_channel_gen+epsilon) , title = "Gen Data Channel Color Distributions")
    

    #ch_px_dist = tfd.Normal(loc=mean_per_channel+epsilon, scale=std_per_channel+epsilon)

    print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))



    # Finally log to Tensorboard
    with summary_writer.as_default(step=epoch):
      # Log Losses
      tf.summary.scalar('gen_loss', avg_gen_loss)
      tf.summary.scalar('disc_loss', avg_disc_loss)
      # ADD partial loss ? 

      # Save image to tensorboard - image saving not working
      #tf.summary.image('image_at_epoch_{:04d}'.format(epoch+1), [stacked_image.astype(np.uint8)])
      # In order to have a timeseries of images - probably
      tf.summary.image('Generated Image', tf.expand_dims(stacked_image , 0) )


      if valid_dataset:
        # Log Accuracy on TB
        tf.summary.scalar('disc_accuracy_on_real', disc_acc_on_real_image)
        tf.summary.scalar('disc_accuracy_on_fake', disc_acc_on_fake_image)

      if distributions is not None:
        # Per distribution log loss
        for key,value in avg_log_loss_res.items():
          tf.summary.scalar("Avg Log Loss {}".format(key), value)
      
      # Hoping it writes
      summary_writer.flush()

    

    # Track Epochs on Checkpoint
    checkpoint_manager.checkpoint.step.assign_add(1)




# For each distribution passed, calculate the log probabilities that the generated images were generated by those distributions
def extract_distribution_log_losses(distributions , generated_images):
  avg_log_loss_res = {}
  
  for  key , dist in distributions.items() :

    if re.search("Pixel",key):
      mean,std = image_utils.get_mean_std_per_pixel(generated_images)
    elif re.search("Channel" , key):
      mean,std = image_utils.get_mean_std_per_channel(generated_images)
    elif re.search("Patch",key):
      mean,std = image_utils.get_mean_std_per_patch(generated_images , patch_shape  = [2,2] , patch_type = 'channel' )
    else:
      raise Exception("No matching found")

    avg_log_loss_res[key] = get_image_sum_log_prob(mean , dist) / (dist.batch_shape[0]*dist.batch_shape[1] ) # Normalize per pixels/patch number

  for key,value in avg_log_loss_res.items():
    print("Avg Log Loss {} : {}".format(key,value) )
  
  return avg_log_loss_res




'''
label smoothing on discriminator

loss = tf.nn.sigmoid_cross_entropy_with_logits(d_on_data, .9) +  tf.nn.sigmoid_cross_entropy_with_logits(d_on_samples, 0.)
'''
#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_BinaryCrossentropy_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy( from_logits=False )

    # For smooting instead of tf.ones_like(real_output) , just put 0.9
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    # Double the loss
    #real_loss = tf.math.multiply(real_loss, 2)
    #print("Loss modified by 2")

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss , real_loss , fake_loss

def generator_BinaryCrossentropy_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)




@tf.function
def train_step(disc_input , noise_dim ,  generator, discriminator , generator_optimizer , discriminator_optimizer , generator_loss , discriminator_loss , gen_extra_inputs = None , BATCH_SIZE = 128):

    # gen_extra_inputs - a flexible way of adding additional inputs to the generator    
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      # Train the Discriminator K times - In here we have access on only 1 batch. Need to relocate if needed
      for k in range(0,1):

        # Sample Noise
        noise = generate_noise( shape = [BATCH_SIZE, noise_dim])
        #print("NOISE SHAPE : ",gen_input.shape )
        #print("LABELS SHAPE : ",gen_extra_inputs.shape )

        # Just to have explicit names and clarity
        gen_input = noise
        # Check for additional extra inputs for the generator outside the noise
        if gen_extra_inputs is not None:
          gen_input =[gen_input , gen_extra_inputs ]
   


        # Generate Images given inputs
        generated_images = generator( inputs = gen_input , training=True)
        # Output of the discriminator based on real images
        real_output = discriminator( disc_input, training=True)

        # Add additional provided inputs
        disc_fake_input = generated_images
        if gen_extra_inputs is not None:
          disc_fake_input =[disc_fake_input , gen_extra_inputs ]

        # Output of the discriminator based on fake images
        fake_output = discriminator( disc_fake_input , training=True)

        # Discriminator loss based on real and fake images
        disc_loss , real_loss , fake_loss = discriminator_loss(real_output, fake_output)

        #print(f"Discriminator real_loss Batch loss { tf.print(tf.math.reduce_mean(real_loss)) }.  Discriminator fake_loss Batch Loss {tf.print(tf.math.reduce_mean(fake_loss))}")
        #tf.print("Discriminator real_loss Batch loss", tf.math.reduce_mean(real_loss) , "Discriminator fake_loss Batch Loss" ,  tf.math.reduce_mean(fake_loss))

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        # Discriminator Backprop
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
      

      # Train the Generator
  
      # Sample Noise
      gen_input = generate_noise( shape = [BATCH_SIZE, noise_dim])
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
      fake_output = discriminator( disc_fake_input , training=True)
      # Generator loss based only on fake images
      gen_loss = generator_loss(fake_output)
      # Generator Gradient
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      # Generator Backprop
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
      
      
      
      


    return tf.math.reduce_mean(gen_loss) , tf.math.reduce_mean(disc_loss) , tf.math.reduce_mean(real_loss) , tf.math.reduce_mean(fake_loss)


# Validation Step
#@tf.function
def valid_step(valid_dataset):


  # Init accuracy variables - PROBABLY NO NEED FOR TF VARIABLE
  disc_acc_on_real_image = tf.Variable(0.0 , trainable=False)
  disc_acc_on_fake_image = tf.Variable(0.0 , trainable=False)

  # Sigmoid function to transform logits out of the model into probabilities
  sigmoid_func = tf.keras.activations.sigmoid

  # Metric used for the validation
  # Each one of them mantain an internal step, in order to be used on several batches
  binary_accuracy_metric_real = tf.keras.metrics.BinaryAccuracy( threshold = 0.5)
  binary_accuracy_metric_fake = tf.keras.metrics.BinaryAccuracy( threshold = 0.5)

  for image_batch , labels_batch in valid_dataset:

    # Sample Noise
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # Generate Images given Noise And Labels
    generated_images = generator( [noise ,labels_batch], training=False)
    # Output of the discriminator based on fake images
    fake_output = discriminator( [generated_images,labels_batch] , training=False)

    # Output of the discriminator based on real images
    real_output = discriminator( [image_batch,labels_batch] , training=False)
    

    # Real output to probabilities -> Images Being Real
    real_output_prob = sigmoid_func(real_output)
    # Fake output to probabilities - > Fake Image Being Real
    fake_output_prob = sigmoid_func(fake_output)
    
    
    

    # if real_output == 1 -> Discriminator good  | if 0 Disc bad

    _ = binary_accuracy_metric_real.update_state( y_true =  tf.ones_like( real_output_prob ) , y_pred =  real_output_prob  )
    disc_acc_on_real_image = binary_accuracy_metric_real.result().numpy()
    

    # if fake_output == 1 -> Discriminator error | if 0 Disc good
      # Reset the internal state, otherwise keep averagin i THINK
    #binary_accuracy_metric_fake.reset_state()

    _ = binary_accuracy_metric_fake.update_state(  y_true =  tf.zeros_like( fake_output_prob ) , y_pred = fake_output_prob  )
    disc_acc_on_fake_image = binary_accuracy_metric_fake.result().numpy()

    #print( " Accuracy on Real Images : {}   Accuracy on Fake Images : {} ".format(disc_acc_on_real_image , disc_acc_on_fake_image ) )


  return disc_acc_on_real_image , disc_acc_on_fake_image


# Given a distribution , get the sum loglikelyhood -> ( return -loglikelyhood)
def get_image_sum_log_prob(image, dist):

    log_prob = dist.log_prob(image)
    return -np.sum(log_prob)


# Generate N*D normal sample noises
def generate_noise( shape = [1,10]):
  return tf.random.normal( shape )

# Restore or Generate new Seed
def get_seed( shape = [1,10] , seed_save_path = None):

  if seed_save_path :
    
    # Need to modify Load/Save
    try:
      seed =  np.load(seed_save_path)
      print("Restoring seed from {}".format(seed_save_path))
      return seed
    except:
      print("Saving new generated seed to {}".format(seed_save_path))
      noise = generate_noise( shape = shape )
      np.save(seed_save_path,noise)
      return noise

  return generate_noise( shape = shape )
    

    

# Target to One Hot - tf dataset transform
def to_one_hot_encode(x_,y_):
  return x_ , tf.one_hot(y_, 10)