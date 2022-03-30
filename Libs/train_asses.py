import tensorflow as tf
import numpy as np
# Tensorflow Dist
import tensorflow_probability as tfp
tfd = tfp.distributions
from collections import deque # Used to have limited arrays

import re #regex

import image_utils
import custom_metrics
import data_utils

import time
# Manipulate the cells output
from IPython import display

import logging
logger = logging.getLogger(__name__)

from numpy.random import randint




# Used to add some noise to zero probabilities - HARD CODED BAD
epsilon = 1e-3


#IN case of switch into iterations

#batches_per_epoch = floor(dataset_size / batch_size)
#total_iterations = batches_per_epoch * total_epochs

def train(dataset, epochs, summary_writer, checkpoint_manager,generator, gen_seed, distributions = None ,valid_dataset = None,  starting_epoch = 0 ,
          image_save_path = None, DEBUG = False, BATCH_SIZE = 128  ,model = None):


#def train(dataset, epochs,summary_writer,checkpoint_manager ,gen_seed, distributions = None ,valid_dataset = None,  starting_epoch = 0 , model_name = 'default_name',
#          image_save_path = None, DEBUG = False, BATCH_SIZE = 128, **kwargs):
          

  total_batches = dataset.cardinality().numpy()

  # Tracks the loss between original data distribution and the generated distribution
  dist_loss_history= {}
  # Track the KL divergence between the two
  kl_divergence_history= {}

  # Keep the last 5 generated images to compare visually
  image_collection_deque = deque( maxlen= 5)

  # [Real Accuracy, Fake Accuracy]
  #discriminator_accuracy = np.empty( (0,2), float)

  for epoch in range(starting_epoch , epochs):
    start = time.time()
    #gen_loss_per_batch = []
    #disc_loss_per_batch = []

    # Discriminator Decomposite Loss
    #disc_real_loss_per_batch = []
    #disc_fake_loss_per_batch = []

    
    
    step = 0

    # Training Step
    for inputs in dataset:
      # Assuming there's no extra inputs to pass to the generator
      extra_inputs = None
      # This means that there's multiple inputs, hoping tha n_inputs != batch_size :D
      if len(inputs) != BATCH_SIZE: # NEED TO CHANGE OR BATCH WITH DIFFERENT SIZE (DROP REMANINDER FALSE ) GOES ERROr
        # Extracting the inputs for the generator
        extra_inputs = inputs[1:]


    

      if extra_inputs:
        model.train_step(inputs, gen_extra_inputs = extra_inputs)
      else:
        model.train_step(inputs)

      

      #gen_loss , disc_loss , real_loss , fake_loss = train_step_infoGAN( inputs , BATCH_SIZE = BATCH_SIZE, **kwargs)                                                                    





      # Store History
      #gen_loss_per_batch.append(gen_loss)
      #disc_loss_per_batch.append(disc_loss)
      #disc_real_loss_per_batch.append(real_loss)
      #disc_fake_loss_per_batch.append(fake_loss)



      
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
    #avg_gen_loss = tf.math.reduce_mean(gen_loss_per_batch)
    #avg_disc_loss = tf.math.reduce_mean(disc_loss_per_batch)
    #avg_disc_real_loss_per_batch = tf.math.reduce_mean(disc_real_loss_per_batch)
    #avg_disc_fake_loss_per_batch = tf.math.reduce_mean(disc_fake_loss_per_batch)

    # Reset Values
    #gen_loss_per_batch = []
    #disc_loss_per_batch = []
    #disc_real_loss_per_batch = []
    #disc_fake_loss_per_batch = []


    
    
    # Save the model and do stuff every x epochs
    if (epoch + 1) % 1 == 0:
      display.clear_output(wait=True)
      
      save_path = checkpoint_manager.save()
      #save_path = checkpoint.save(file_prefix = 'ckpt')
      print("Saved checkpoint for step {}: {}\n".format(int(checkpoint_manager.checkpoint.step), save_path))
      #print("Generator Epoch loss {:1.3f}".format(avg_gen_loss.numpy())  )
      #print("Discriminator Epoch Total Loss {:1.3f}  Discriminator Epoch Real Loss {:1.3f}  Discriminator Epoch Fake Loss {:1.3f}\n".format(avg_disc_loss.numpy() ,avg_disc_real_loss_per_batch.numpy() , avg_disc_fake_loss_per_batch.numpy()) )

      # Print Metric Results 
      for gan_metric in model.gan_metrics:
        gan_metric.print_results()
        # Reset the state for the next epoch
        gan_metric.reset_state()
            
      # Blank line for order after metrics
      print()
      
  

    # Produce images for the GIF as you go   
      # Save Image every x epochs - Due to Memory Constraint
    save_image_condition = (epoch + 1) % 5 == 0
    image_name = 'image_at_epoch_{:04d}.png'.format(epoch+1)
    save_param = {"Save Image" : save_image_condition, "Path" : image_save_path, "filename" : image_name}
    
    # Generate Images From The Seed
    generated_seed_images = generator(gen_seed, training=False)
    denorm_generated_seed_images = image_utils.denormalize_images(generated_seed_images)


    
    # Compute the log losses of the generated images vs original images based on the distributions
    if distributions is not None:
      avg_log_losses, avg_kl_divergence = extract_distribution_losses(distributions, denorm_generated_seed_images)
      
      # Mantain an history
      # Per distribution loss
      for key, value in avg_log_losses.items():
        new_key = "Log loss on" + key
        # Initialize dict
        if new_key not in dist_loss_history:
          dist_loss_history[new_key] = []
        # Store value
        dist_loss_history[new_key].append(value)

      # Per distribution KL div
      for key, value in avg_kl_divergence.items():
        new_key = "KL div on" + key
        # Initialize dict
        if new_key not in kl_divergence_history:
          kl_divergence_history[new_key] = []
        # Store value
        kl_divergence_history[new_key].append(value)



    # Display and Save the Generated Image
    image_utils.display_multiple_image(denorm_generated_seed_images , size = (460,460) , inline=False , save_param = save_param , denormalize = False )

    
    # Stack multiple images generated into a single one
    stacked_image = image_utils.stack_multiple_images(denorm_generated_seed_images)

    # Keep Track Of latest Generations
    image_collection_deque.append(stacked_image)
    # Display latest generated images
    image_utils.display_multiple_image(image_collection_deque , size = (600,600) , inline=True  )
        


    
    #Extract Distribution from Generated images
    gen_distribution = image_utils.extract_distribution(denorm_generated_seed_images, of_type = "channel", epsilon = 1e-3)

    # Plot the Real vs Generated Distribution
    image_utils.plot_channels_dist( [ distributions['Distribution Per Channel'], gen_distribution], image_channels = 1, title= [ "Original Data Color Distribution Per Channel", "Gen Data Color Distribution Per Channel"] )
    # ADD the rest of the distribution TO PLOT

    # Plot the Metrics
    custom_metrics.plot_metrics(model.gan_metrics)

    # Plot the Log losses and The KL Divergences
    if distributions:
      data_utils.plot_data_from_dicts_list( [dist_loss_history, kl_divergence_history] )
      
        


    print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))



    # Finally log to Tensorboard
    with summary_writer.as_default(step=epoch):
      # Log Losses
      #tf.summary.scalar('gen_loss', avg_gen_loss)
      #tf.summary.scalar('disc_loss', avg_disc_loss)
      # ADD partial loss ? 

      # Save image to tensorboard
      if ( epoch+ 1) % 5 == 0 :
        # In order to have a timeseries of images , the name has to be the same
        tf.summary.image('Generated Image', tf.expand_dims(stacked_image , 0) )

      # Store metrics
      for gan_metric in model.gan_metrics:
        if gan_metric.GAN_module == 'discriminator':
          real_metric_data, fake_metric_data=  gan_metric.get_history()
          # Store to tensoardboard the latest value
          tf.summary.scalar("{} on Real Data".format(gan_metric.name), real_metric_data[-1])
          tf.summary.scalar("{} on Gen Data".format(gan_metric.name), fake_metric_data[-1])

        #elif gan_metric.GAN_module == 'generator':
        else:
          metric_data =  gan_metric.get_history()
          tf.summary.scalar("{}".format(gan_metric.name), metric_data[-1])

        #else : raise Exception("GAN_module not recognized. Must be 'discriminator' or 'generator'")
  


      if distributions is not None:
        # Per distribution log loss
        for key, value in avg_log_losses.items():
          tf.summary.scalar("Avg Log Loss on {}".format(key), value)
        
        # Per distribution KL div
        for key, value in avg_kl_divergence.items():
          tf.summary.scalar("Avg KL Divergence on {}".format(key), value)


      if valid_dataset:
        # Log Accuracy on TB
        tf.summary.scalar('disc_accuracy_on_real', disc_acc_on_real_image)
        tf.summary.scalar('disc_accuracy_on_fake', disc_acc_on_fake_image)  
      
      # Hoping it writes
      summary_writer.flush()

    

    # Track Epochs on Checkpoint
    checkpoint_manager.checkpoint.step.assign_add(1)








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







# For each distribution passed, calculate the log probabilities that the generated images were generated by those distributions and the KL divergence
def extract_distribution_losses(distributions , generated_images, log10= True):
  avg_log_losses = {}
  avg_kl_divergences = {}
  
  # Parse the distributions and extract the correct one from the generated images
  for  key , dist in distributions.items() :

    # Distribution per Pixel
    if re.search("Pixel",key):
      mean, std = image_utils.get_mean_std_per_pixel(generated_images)
    # Distribution per Channel
    elif re.search("Channel" , key):
      mean, std = image_utils.get_mean_std_per_channel(generated_images)
    # Distribution per Patch
    elif re.search("Patch",key):
      mean, std = image_utils.get_mean_std_per_patch(generated_images , patch_shape  = [2,2] , patch_type = 'channel' )
    else:
      raise Exception("No matching found")

    # Distribution of the generated images
    gen_dist = tfd.Normal(loc= tf.dtypes.cast(mean, tf.float32)+ epsilon, scale=tf.dtypes.cast(std, tf.float32)+ epsilon)
    #gen_dist = tfd.Normal(loc= mean + epsilon, scale=std+ epsilon)
    # KL divergence
    mean_kl = mean_kl_divergence(dist, gen_dist)

    # Store Results
    avg_kl_divergences[key] = mean_kl
    avg_log_losses[key] = get_image_sum_log_prob(mean , dist) / (dist.batch_shape[0]*dist.batch_shape[1] ) # Normalize per pixels/patch number

    # Scale
    if log10:
      avg_kl_divergences[key] = np.log10(avg_kl_divergences[key])
      avg_log_losses[key] = np.log10(avg_log_losses[key])



  # MOVE THE DISPLAY INTO THE MAIN AREA
  # Display Results
  for key,value in avg_log_losses.items():
    print("Avg Log Loss on {} : {}".format(key,value) )
  
  print()

  # Display Results
  for key,value in avg_kl_divergences.items():
    print("Avg KL Divergence on {} : {}".format(key,value) )
  
  return avg_log_losses, avg_kl_divergences








# Compute the mean of the KL divergence per pixel
def mean_kl_divergence(dist_a, dist_b):
    kl = tfp.distributions.kl_divergence(dist_a, dist_b, allow_nan_stats=True, name=None)

    return np.mean(kl)


# Given a distribution , get the sum loglikelyhood -> ( return -loglikelyhood)
def get_image_sum_log_prob(image, dist):

    log_prob = dist.log_prob(image)
    return -np.sum(log_prob)



# Source Weight and Biases
# calculate frechet inception distance
def calculate_fid(real_embeddings, generated_embeddings):
  # calculate mean and covariance statistics
  #mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
  #mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
  
  mu1, sigma1 = np.mean(real_embeddings, axis= 0), np.cov(real_embeddings, rowvar=False)
  mu2, sigma2 = np.mean(generated_embeddings, axis= 0), np.cov(generated_embeddings,  rowvar=False)

  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  from scipy.linalg import sqrtm
  # Matrix square root
  covmean = sqrtm(sigma1.dot(sigma2))
  #covmean = np.sqrt(sigma1.dot(sigma2))

  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
      covmean = covmean.real
  # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

  return fid




# Restore or Generate new Seed for the Generator
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

  return generate_noise(shape = shape )

# Restore or Generate new Seed for the Generator
def get_infogan_seed( shape, n_class, seed_save_path = None):

  if seed_save_path :
    
    # Need to modify Load/Save
    try:
      seed =  np.load(seed_save_path)
      print("Restoring seed from {}".format(seed_save_path))
      return seed
    except:
      print("Saving new generated seed to {}".format(seed_save_path))
      noise = tf.keras.layers.Concatenate()( (sample_infogan_gen_input(batch_size=shape[0], noise_dim=shape[1], n_class=n_class)) )
      np.save(seed_save_path,noise)
      return noise

  return generate_noise(shape = shape )

# Generate or load a conditioning seed
def get_cond_seed(num_examples, n_class,as_one_hot = False, seed_save_path= None):

    # Check if a seed already exists
    if seed_save_path :
      # Need to modify Load/Save
      try:
        cond_seed =  np.load(seed_save_path)
        print("Restoring seed from {}".format(seed_save_path))
        return cond_seed
      except:
        print("Generating and storing new generated cond seed to {}".format(seed_save_path))



    if n_class > num_examples:
      raise Exception("atleast a number of examples equal to the classes has to be generated")
    
    if num_examples % n_class:
      raise Exception("The num_examples is not a multiple of the number of class")
      
    
    # Final Array
    cond_seed = np.zeros(shape= [num_examples, n_class if as_one_hot else 1 ], dtype= np.int32 )

    # Indexing
    offset= int(num_examples/ n_class) # how many examples ofthe same class


    #print(f"Batch {num_examples} , classes {n_class}, offset {offset}")

    # One hot encode input
    if as_one_hot:  
      for i in range(n_class):
          sample_target = [0] * n_class
          # Set the index categorical value
          sample_target[i] = 1
          cond_seed[i*offset:(i+1)*offset] = sample_target # sample * 4 istance
    # Categorical input
    else:
      for i in range(n_class):
        # Clone the sample n(4) times
        cond_seed[i*offset:(i+1)*offset] = i # sample * 4 istance
    
    # Save the generated seed
    if seed_save_path :
      np.save(seed_save_path, cond_seed)

    return cond_seed



# Generate N*D normal sample noises
def generate_noise( shape = [1,10]):
  return tf.random.normal( shape )    

# Target to One Hot - tf dataset transform
def to_one_hot_encode(x_, y_, n_classes = 10):
  return x_, tf.one_hot(y_, n_classes)


# Info Gan input
def sample_infogan_gen_input(batch_size=32, noise_dim=62, n_class=10, seed=None):
  # create noise input
  noise = tf.random.normal([batch_size, noise_dim], seed=seed)
  # Create categorical latent code
  label = tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.int32, seed=seed)
  label = tf.one_hot(label, depth=n_class)
  # Create one continuous latent code
  c_1 = tf.random.uniform([batch_size, 1], minval=-1, maxval=1, seed=seed)

  
  return label, c_1, noise
