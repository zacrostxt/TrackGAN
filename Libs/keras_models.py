import tensorflow as tf
#import numpy as np


#Generator architecture
def make_generator_model(input_shape = 100):


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(6*6*512, use_bias=False, input_shape=(input_shape,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Reshape((6, 6, 512)))
    
    #8
    model.add(tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #16
    model.add(tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #18
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #20
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #22
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

    #44
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

    #46
    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

   


    model.add(tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='valid', use_bias=True, activation='tanh'))

    # Smooth the image
    #model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2) , strides = (1,1) , padding='valid' )  )



    #assert model.output_shape == (None, 48, 48, 1)

    return model
  
def make_discriminator_model(input_shape = [48, 48, 3]):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid ) )

    return model











#Generator architecture - CHANGE TO RELU
def make_conditioned_generator_model(input_shape_noise = 100 , input_shape_conditioning = 10):

    
    # Zoise path
    noise_input = tf.keras.layers.Input(shape=(input_shape_noise) , name ='z_input')
    z_dense = tf.keras.layers.Dense(7*7*50, use_bias=True)(noise_input)
    z_batch = tf.keras.layers.BatchNormalization()(z_dense)
    z_lrelu = tf.keras.layers.LeakyReLU()(z_batch)

    # Conditioning Input path
    c_input = tf.keras.layers.Input(shape=(input_shape_conditioning), name ='c_input')
    c_dense = tf.keras.layers.Dense(196)(c_input)
    #c_batch = tf.keras.layers.BatchNormalization()(c_dense)
    c_lrelu =  tf.keras.layers.LeakyReLU()(c_dense)
    
    #2450 + 98 = 2548 = 7x7x52
    #1620 + 162 = 1782 = 9x9x2

    # 7x7x30 = 1470
    # 7, 7, 50 = 2450
    # 7x7x54 = 2646



    # Merge inputs paths
    concat_layer = tf.keras.layers.concatenate([z_lrelu, c_lrelu])

    x = concat_layer
    

    layers_list = [tf.keras.layers.Reshape((7, 7, 54)),
                   tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid', use_bias=True),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid', use_bias=True),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='valid', use_bias=True),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=True),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='valid', use_bias=True, activation='tanh')
                   ]

    for layer in layers_list:
      x = layer(x)

    

    model = tf.keras.Model(inputs=[noise_input , c_input], outputs=x)   


    #assert model.output_shape == (None, 28, 28, 1)

    return model







def make_conditioned_discriminator_model(input_image_shape = [28, 28, 1],input_shape_conditioning = 10):


   # Image path
    image_input = tf.keras.layers.Input(shape=(input_image_shape) , name ='image_input')
    image_path = image_input

    # Conv -> Leaky -> Dropout x2
    for filters , stride in zip([32,32,64,128] , [ 1 ,1, 2 ,2 ]):
      image_path = tf.keras.layers.Conv2D(filters, (3, 3), strides=(stride, stride), padding='valid')(image_path)
      image_path = tf.keras.layers.BatchNormalization()(image_path) # DCGAN
      image_path = tf.keras.layers.LeakyReLU()(image_path)
      #image_path = tf.keras.layers.Dropout(0.3)(image_path)
      
    image_path = tf.keras.layers.Flatten()(image_path)


    # Conditioning Input path
    c_input = tf.keras.layers.Input(shape=(input_shape_conditioning), name ='c_input')
    c_dense = tf.keras.layers.Dense(196)(c_input)
    #c_batch = tf.keras.layers.BatchNormalization()(c_dense)
    c_lrelu =  tf.keras.layers.LeakyReLU()(c_dense)
    

    # Merge inputs paths
    concat_layer = tf.keras.layers.concatenate([image_path, c_lrelu])

    #print(concat_layer.shape)

    x = concat_layer
    
    # Common Path
    common_dense = tf.keras.layers.Dense(50)
    # Maybe droout around here
    common_dense_1 = tf.keras.layers.Dense(30)
    out = tf.keras.layers.Dense(1 , activation = tf.keras.activations.sigmoid) 

    x = common_dense(x)
    #x = common_dense_1(x)
    out = out(x)


    model = tf.keras.Model(inputs=[image_input , c_input], outputs = out)    

    return model














'''
THE FIRST ONE

#Generator architecture
def make_generator_model(input_shape = 100):


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(6*6*256, use_bias=False, input_shape=(input_shape,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((6, 6, 256)))
    

    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    #assert model.output_shape == (None, 48, 48, 1)

    return model
  
def make_discriminator_model(input_shape = [48, 48, 3]):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model















#4M 1M
#Generator architecture
def make_generator_model(input_shape = 100):


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(6*6*512, use_bias=False, input_shape=(input_shape,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Reshape((6, 6, 512)))
    

    model.add(tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )


    model.add(tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=True, activation='tanh'))

    #assert model.output_shape == (None, 48, 48, 1)

    return model
  
def make_discriminator_model(input_shape = [48, 48, 3]):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid ) )

    return model
































    #Generator architecture CONDITIONED
def make_conditioned_generator_model(input_shape_noise = 100 , input_shape_conditioning = 10):

    
    # Zoise path
    noise_input = tf.keras.layers.Input(shape=(input_shape_noise) , name ='z_input')
    z_dense = tf.keras.layers.Dense(7*7*20, use_bias=False)(noise_input)
    z_batch = tf.keras.layers.BatchNormalization()(z_dense)
    z_lrelu = tf.keras.layers.LeakyReLU()(z_batch)

    # Conditioning Input path
    c_input = tf.keras.layers.Input(shape=(input_shape_conditioning), name ='c_input')
    c_dense = tf.keras.layers.Dense(98)(c_input)
    #c_batch = tf.keras.layers.BatchNormalization()(c_dense)
    c_lrelu =  tf.keras.layers.LeakyReLU()(c_dense)
    
    #2450 + 98 = 2548 = 7x7x52
    #1620 + 162 = 1782 = 9x9x2

    # 500 + 50 = 550

    # 980 + = 1078


    # Merge inputs paths
    concat_layer = tf.keras.layers.concatenate([z_lrelu, c_lrelu])

    x = concat_layer
    

    layers_list = [tf.keras.layers.Reshape((7, 7, 22)),
                   tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(),
                   tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
                   ]

    for layer in layers_list:
      x = layer(x)

    

    model = tf.keras.Model(inputs=[noise_input , c_input], outputs=x)   


    assert model.output_shape == (None, 28, 28, 1)

    return model
'''