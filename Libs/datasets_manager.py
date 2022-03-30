import numpy as np
import tensorflow as tf


#0-9 digits dataset
def mnist_digit_dataset():
  from keras.datasets import mnist


  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  X_ = np.concatenate((x_train, x_test), axis=0)
  Y_ = np.concatenate((y_train, y_test), axis=0)

  X_ = X_.reshape(X_.shape[0], 28, 28, 1).astype('float32')

  
  print( f"Input Shape : {X_.shape} \nOutput Shape : {Y_.shape}" )
  

  return X_, Y_


def dummy_RGB_dataset():
  image = np.zeros( shape = [28,28,3])

  X_ = np.zeros( shape = [90,28,28,3] )
  Y_ = np.zeros( shape = [90,1] )


  # RED IMAGES
  X_[:30,2:14,2:14,0] = tf.random.normal( X_[:30,2:14,2:14,0].shape, 160, 10, tf.float32)
  Y_[:30] = 0
  # GREEN
  X_[30:60,0:14,14:28,1] = tf.random.normal( X_[30:60,0:14,14:28,1].shape, 160, 10, tf.float32)
  Y_[30:60] = 1
  # BLUE
  X_[60:90,10:20,10:20,2] = tf.random.normal( X_[60:90,10:20,10:20,2].shape, 160, 10, tf.float32)
  Y_[60:90] = 2

  print( f"Input Shape : {X_.shape} \nOutput Shape : {Y_.shape}" )
  
  return X_, Y_

def dummy_RGB_dataset_transformed():

  # Trasform data
  trasformations = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomTranslation(height_factor = 0.05, width_factor=0.05),
  ])
  

  X_, Y_ = dummy_RGB_dataset()

  X_ = trasformations(X_)

  return X_, Y_