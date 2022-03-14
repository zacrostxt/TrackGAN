import tensorflow as tf
import numpy as np

import logging

logger = logging.getLogger(__name__)


def test():
  logger.debug('This is my 😂 debug message ')
  return


#!pip install scikit-learn==0.20
#from sklearn.externals import joblib
#X = load_train(base_path + '/DATA_Rep/pixelart-48x48.dump')
#save_path = base_path + 'DATA_Rep/pixelart-48x48.npy'
#np.save(save_path, X)

def load_train(filename, dim_ordering = 'tf', debug = True, bgr_to_rgb = False) :
    # Expects files to be loaded and pickled as RGB image 
    X = joblib.load(filename)['X']

    if debug :
        print("Color range pre: ", np.min(X), np.max(X) )


    # Scale color range from (0, 1) to (-1, 1)
    X = X * 2 - 1
    assert np.min(X) >= -1.0 and np.max(X) <= 1.0
    if debug :
        print("Color range: ", np.min(X), np.max(X) )
    
    if bgr_to_rgb :
        X = X[:, :, :, [2, 1, 0]]
    
    assert X.shape[-1] == 3
    if dim_ordering == 'th' :
        # Change dim_ordering from TF (count_images, height, width, 3) to theano (count_images, 3, height, width)
        X = np.swapaxes(X, 2, 3)
        X = np.swapaxes(X, 1, 2)
    elif dim_ordering == 'tf' :
        pass
    else :
        raise ValueError('Dim ordering %s is not supported' % dim_ordering)
    
    if debug :
        print('Shape:', X.shape)
    
    return X