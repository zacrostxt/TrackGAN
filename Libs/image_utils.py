import numpy as np
import tensorflow as tf
import os

# Libs for Images
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import glob #it does pattern matching and expansion <- Retrieving filenames on system and such






# Plot the histogram per pixel channel mean and the scatter plot of the sampled data on the x-axis and the prob associated in the y-axis
def plot_channels_dist(dist, title = None):
  data = dist.sample(10)
  
  # Extract the mean value per pixel per channel
  avg_data = np.mean(data,axis = 0 ) # used for the histogram
  # Size of the plot
  plt.subplots(figsize=( 15, 5) )

  channels = ['red','green','blue']
  # Grayscale image
  if data.shape[3] == 1:
    channels = [ ['orange'] ]

  for ch in range(data.shape[3]):

    #Scatter plot
    plt.subplot(2 ,  3 , ch+1 )
    plt.subplots_adjust()
    plt.scatter(data[:, :,:,ch],  dist.prob(data)[:, :,:,ch] , color=channels[ch], alpha=0.4)

    # Histogram
    plt.subplot(2 ,  3 , ch+1+3 )
    plt.hist( avg_data[:,:,ch]) 


  # Add title
  if title is not None:
    plt.title(title)
  plt.show()

  return




  # Channel selector
  i = 0
  # Do it for each channel
  for color in ['red','green','blue']:

    #Scatter plot
    plt.subplot(2 ,  3 , i+1 )
    plt.subplots_adjust()
    plt.scatter(data[:, :,:,i],  dist.prob(data)[:, :,:,i] , color=color, alpha=0.4)

    # Histogram
    plt.subplot(2 ,  3 , i+1+3 )
    plt.hist( avg_data[:,:,i]) 

    i+=1
  


def get_mean_std_per_pixel(image_dataset):
  
  assert image_dataset.shape[-1] == 3 or image_dataset.shape[-1] == 1

  sum_across_images = np.sum(image_dataset,axis=0)
  #print("Summed across images. New shape : \n",sum_across_images.shape)

  mean_across_channels = np.mean(sum_across_images,axis =-1) #mean per pixel
  std_across_channels = np.std(sum_across_images,axis =-1)  #std per pixel

  #print("Mean across channels. New shape : \n",mean_across_channels.shape)
  #print("STD across channels. New shape : \n",std_across_channels.shape)


  return mean_across_channels, std_across_channels


def get_mean_std_per_channel(image_dataset):
  
  assert image_dataset.shape[-1] == 3 or image_dataset.shape[-1] == 1

  mean_per_channel = np.mean(image_dataset,axis = 0)
  std_per_channel = np.std(image_dataset,axis = 0)

  #print("Mean per channel . New shape : \n",mean_per_channel.shape)
  #print("Std per channel . New shape : \n",std_per_channel.shape)

  return mean_per_channel,std_per_channel


# Compute the average of a small window on the images
def get_mean_std_per_patch(image_dataset , patch_shape  = (2,2) , patch_type = 'channel' , overlapping = False):
    
    strides = patch_shape
    # Patches overlaps on same pixels
    if overlapping :
      strides = (1,1)
    
    
    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size = patch_shape, strides= strides , padding='valid')

    # Channel Wise Patch
    if patch_type == 'channel':
      mean, std = get_mean_std_per_channel(image_dataset)
    # Pixel Wise Patch
    else:
      mean, std = get_mean_std_per_pixel(image_dataset)
      mean = tf.reshape( mean , shape =[mean.shape[0] , mean.shape[1] , 1])
      std = tf.reshape( std , shape =[std.shape[0] , std.shape[1] , 1])
    
    mean = avg_pool_2d( tf.expand_dims(mean , axis = 0) )
    std = avg_pool_2d( tf.expand_dims(std , axis = 0) )

    # Eliminate the batch size
    return mean[0], std[0]







def display_multiple_image(images_array , inline = True ,  size = [300,300] , save_param = {"Save Image" : False,"Path" : None, "filename" : "img.jpg"} , denormalize = False):
  
  
  # N images
  n_samples = len(images_array)
  # Shape of each image
  image_shape = images_array[0].shape
  
  # Shape of matplot subplots
  shape = [1,n_samples]
  if not inline:
    shape = ( int( np.sqrt(n_samples)) , int( np.sqrt(n_samples) ) )


  #fig = plt.figure(figsize=figsize)
  px = 1/plt.rcParams['figure.dpi']  # inches to pixel conversion
  #plt.subplots(figsize=( image_shape[0]*modify_size_by*px, image_shape[1]*modify_size_by*px) )
  _ = plt.figure(figsize=( size[0]*px, size[1]*px))
  # Eliminate all the padding/margin -> 1px of
  plt.subplots_adjust(wspace=1*px, hspace=1*px)

  for i in range( n_samples ):
      plt.subplot(shape[0], shape[1] , i+1)

      image = images_array[i]

      if denormalize:
         image = denormalize_image(image)

      # The standard one
      cmap = 'viridis'
      # In order to plot [w,h,1] images
      if image.shape[2] == 1:
        image = image[:, :, 0]
        cmap = 'gray'

      plt.imshow(image, cmap = cmap)
      plt.axis('off')
  
  if save_param["Save Image"]:
    # Check or Create Path
    if not os.path.exists(save_param["Path"]):
      os.makedirs(save_param["Path"])
    # Save Image to File
    plt.savefig(save_param["Path"] + save_param["filename"])
      
  
  plt.show()



# Display image on cell
def display_image(image_array, size = [100,100]):
  px = 1/plt.rcParams['figure.dpi']  # inches to pixel conversion

  # Modify figure size
  _ = plt.figure(figsize=( size[0]*px, size[1]*px))
  plt.axis('off')

  # Tensorflow related assignment
  image = image_array

  # The standard one
  cmap = 'viridis'
  # In order to plot [w,h,1] images
  if image_array.shape[2] == 1:
    image = image[:, :, 0]
    cmap = 'gray'

  plt.imshow(image_array,cmap = cmap)
  plt.show()



# Stack multiple images togheter by row/column
def stack_multiple_images(images_array , inline = False):
  n_images = len(images_array)

  # Square size of images
  size = int(np.sqrt(n_images))

  image_vertical_slices = []
  # First stack vertically
  for i in range(0, n_images ,size):
    new_image = np.concatenate((images_array[i:i+size]) , axis =0)
    image_vertical_slices.append(new_image)

  stacked_image = None
  # Stack Vertical slices horizontally
  for image_slice in image_vertical_slices:
    if stacked_image is None:
      stacked_image = image_slice
    else:
      stacked_image = np.concatenate((stacked_image , image_slice) , axis = 1)

  return stacked_image


# Save image to File Using PIL
def save_image(image_array , path_to_file):
  image = Image.fromarray(np.uint8( image_array))
  image.save(path_to_file)

# Generate A GIF image using image.io
def generate_GIF(image_source_path, dest_file):

  with imageio.get_writer(dest_file, mode='I') as writer:
    
    filenames = glob.glob(image_source_path + 'image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


def normalize_image(x_,y_): 
  
  #x_ = tf.cast(x_, tf.float32)
  x_ = (x_-127.5)/127.5

  return x_,  y_

def normalize_images(images_arr):
  return (images_arr-127.5)/127.5

def denormalize_images(images_arr):
  return tf.cast( (images_arr*127.5 + 127.5) , tf.uint8)
  
  #return (images_arr*127.5 + 127.5).astype(int)
  

def denormalize_image(image):
  image = np.array((image*127.5 + 127.5) ).astype(int)

  return tf.cast( image , tf.uint8)