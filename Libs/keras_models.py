import tensorflow as tf
#import numpy as np

import logging
logger = logging.getLogger(__name__)

# End to End Gan
def define_gan(g_model, d_model):
  # make the discriminator layer as non trainable
  d_model.trainable = False
  # get the noise and label input from the generator
  gen_noise, gen_label = g_model.input
  # get the output from the generator
  gen_output = g_model.output
  #connect image output and label input from generator as inputs to      #discriminator
  gan_output = d_model([gen_output,gen_label])
  #define gan model as taking noise and label and outputting a #classification
  model = tf.kreas.Model([gen_noise,gen_label],gan_output)
 
  return model


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









def define_generator(latent_dim, n_class):
  label_input = tf.keras.layers.Input(shape=(1,))
  #Embedding layer
  em = tf.keras.layers.Embedding(n_class,20)(label_input)
  nodes = 7*7
 
  em = tf.keras.layers.Dense(nodes)(em)
  em = tf.keras.layers.Reshape((7,7,1))(em)

  #image generator input
  image_input = tf.keras.layers.Input(shape=(latent_dim,))
  nodes = 20*7*7
  d1 = tf.keras.layers.Dense(nodes)(image_input)
  d1 = tf.keras.layers.LeakyReLU(0.2)(d1)
  d1 = tf.keras.layers.Reshape((7,7,20))(d1)
  
  # merge
  merge = tf.keras.layers.Concatenate()([d1,em])
  # Extra layer
  merge = tf.keras.layers.Conv2DTranspose( 128, (3,3), strides=(1,1), padding='same')(merge)
  merge = tf.keras.layers.Conv2DTranspose( 128, (3,3), strides=(1,1), padding='same')(merge)
  #upsample to 14x14
  gen = tf.keras.layers.Conv2DTranspose( 64, (3,3), strides=(2,2), padding='same')(merge)
  gen = tf.keras.layers.LeakyReLU(0.2)(gen)
  #upsample to 28x28
  gen = tf.keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2), padding='same')(gen)
  gen = tf.keras.layers.LeakyReLU(0.2)(gen)
  # Extra
  gen = tf.keras.layers.Conv2DTranspose(16,(3,3),strides=(1,1), padding='same')(gen)
  gen = tf.keras.layers.LeakyReLU(0.2)(gen)
  #output layer 
  out_layer = tf.keras.layers.Conv2D(3,(3,3),activation='tanh', padding='same')(gen)
  #define model 
  model = tf.keras.Model([image_input,label_input],out_layer)
  return model







def define_discriminator_rgb(n_class, input_shape=(28,28,3)):
  # label input
  in_labels = tf.keras.layers.Input(shape=(1,))
  # Embedding for categorical input
  em = tf.keras.layers.Embedding(n_class,20)(in_labels)

  # scale up the image dimension with linear activations
  d1 = tf.keras.layers.Dense(input_shape[0] * input_shape[1] * input_shape[2])(em)
  # reshape to additional channel
  d1 = tf.keras.layers.Reshape((input_shape[0],input_shape[1],input_shape[2] ))(d1)


  # image input
  image_input = tf.keras.layers.Input(shape=input_shape)

  #  concate label as channel
  merge = tf.keras.layers.Concatenate()([image_input,d1])
  # downsample
  fe = tf.keras.layers.Conv2D(128,(3,3),strides=(2,2),padding='same')(merge)
  fe = tf.keras.layers.LeakyReLU(0.2)(fe)
  # downsample
  fe = tf.keras.layers.Conv2D(64,(3,3),strides=(2,2),padding='same')(merge)
  fe = tf.keras.layers.LeakyReLU(0.2)(fe)
  # downsample
  fe = tf.keras.layers.Conv2D(32,(3,3),strides=(2,2),padding='same')(merge)
  fe = tf.keras.layers.LeakyReLU(0.2)(fe)
  #flatten feature maps
  fe = tf.keras.layers.Flatten()(fe)
  fe = tf.keras.layers.Dropout(0.4)(fe)
  #ouput
  out_layer = tf.keras.layers.Dense(1,activation='sigmoid')(fe)
  #define model
  model = tf.keras.Model([image_input,in_labels],out_layer)
  
  
  return model








# INFO GAN MODEL
def infogan_generator_continuous(n_filters=128, input_size=73):
    # Build functional API model
    # input
    input = tf.keras.layers.Input(shape=(input_size, ))

    # Fully-connected layer.
    dense_1 = tf.keras.layers.Dense(units=64, use_bias=False) (input)
    bn_1 = tf.keras.layers.BatchNormalization()(dense_1)
    act_1 = tf.keras.layers.ReLU()(bn_1)

    # Fully-connected layer. The output should be able to reshape into 7x7
    dense_2 = tf.keras.layers.Dense(units=7*7*20, use_bias=False) (act_1)
    bn_2 = tf.keras.layers.BatchNormalization()(dense_2)
    act_2 = tf.keras.layers.ReLU()(bn_2)

    # Reshape
    reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 20))(act_2)

    nf = n_filters
    # First transposed convolutional layer

    tc_1 = tf.keras.layers.Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(reshape)
    bn_1 = tf.keras.layers.BatchNormalization()(tc_1)
    act_1 = tf.keras.layers.ReLU()(bn_1)

    # Number of filters halved after each transposed convolutional layer
    nf = nf//2
    # Second transposed convolutional layer
    # strides=(2, 2): shape is doubled after the transposed convolution
    tc_2 = tf.keras.layers.Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(act_1)
    bn_2 = tf.keras.layers.BatchNormalization()(tc_2)
    act_2 = tf.keras.layers.ReLU()(bn_2)

    # Number of filters halved after each transposed convolutional layer
    nf = nf//2
    # Second transposed convolutional layer
    # strides=(2, 2): shape is doubled after the transposed convolution
    tc_3 = tf.keras.layers.Conv2DTranspose(nf, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(act_2)
    bn_3 = tf.keras.layers.BatchNormalization()(tc_3)
    act_3 = tf.keras.layers.ReLU()(bn_3)



    # Final transposed convolutional layer: output shape: 28x28x1, tanh activation
    output = tf.keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), 
                                         padding="same", activation="tanh")(act_3)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    
    return model


def infogan_discriminator_continuous(n_filters=64, n_class=10, input_shape=(28, 28, 1)):
    # Build functional API model
    # Image Input
    image_input = tf.keras.layers.Input(shape=input_shape)

    nf = n_filters
    c_1 = tf.keras.layers.Conv2D(nf, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=True)(image_input)
    bn_1 = tf.keras.layers.BatchNormalization()(c_1)
    act_1 = tf.keras.layers.LeakyReLU(alpha=0.1)(bn_1)

    
    
    # Second convolutional layer
    # Output shape: 7x7
    c_2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False)(act_1)
    bn_2 = tf.keras.layers.BatchNormalization()(c_2)
    act_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(bn_2)

    # Number of filters doubled after each convolutional layer
    nf = nf*2
    # Third convolutional layer
    # Output shape: 7x7
    c_3 = tf.keras.layers.Conv2D(100, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False)(act_2)
    bn_3 = tf.keras.layers.BatchNormalization()(c_3)
    act_3 = tf.keras.layers.LeakyReLU(alpha=0.1)(bn_3)

    # Flatten the convolutional layers
    flatten = tf.keras.layers.Flatten()(act_3)

    # FC layer
    dense = tf.keras.layers.Dense(12, use_bias=False)(flatten)
    bn = tf.keras.layers.BatchNormalization()(dense)
    act = tf.keras.layers.LeakyReLU(alpha=0.1)(bn)
    # Discriminator output. Sigmoid activation function to classify "True" or "False"
    d_output = tf.keras.layers.Dense(1, activation='sigmoid')(act)

    # Auxiliary output. 
    q_dense = tf.keras.layers.Dense(12, use_bias=False)(act)
    q_bn = tf.keras.layers.BatchNormalization()(q_dense)
    q_act = tf.keras.layers.LeakyReLU(alpha=0.1)(q_bn)

    # Classification (discrete output)
    clf_out = tf.keras.layers.Dense(n_class, activation="softmax")(q_act)

    # Gaussian distribution mean (continuous output)
    mu = tf.keras.layers.Dense(1)(q_act)

    # Gaussian distribution standard deviation (exponential activation to ensure the value is positive)
    sigma = tf.keras.layers.Dense(1, activation=lambda x: tf.math.exp(x))(q_act)

    # Discriminator model (not compiled)
    d_model = tf.keras.models.Model(inputs=image_input, outputs=d_output)

    # Auxiliary model (not compiled)
    q_model = tf.keras.models.Model(inputs=image_input, outputs=[clf_out, mu, sigma])

    return d_model, q_model












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