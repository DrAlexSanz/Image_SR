import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os, shutil
from distutils.dir_util import copy_tree
import tensorflow as tf


import numpy as np
import os, shutil
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Concatenate, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import Adam, Adadelta, Adagrad
from keras.callbacks import History, ModelCheckpoint
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image, ImageFilter

K.set_image_data_format("channels_last")



def small_UNET():
    """
    This function returns a keras model instance. Compile and fit is done outside.
    """
    
    input_image = Input(shape = (1, 28, 28)) # Change later.
    
    X1 = Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu", padding = "same")(input_image)
    
    X2 = Conv2D(filters = 64, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same")(X1)
    
    X2 = BatchNormalization()(X2)
        
    encoded = Conv2D(filters = 64, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same")(X2)
    
    
    X3 = UpSampling2D((2, 2))(encoded)
    
    X4 = Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu", padding = "same")(X3)
    
#     X4 = Add()([X4, X2])
    # Don't add like in a ResNet! This is a convnet, if I add I am feeding info, as I want, but it's kind of hidden by the addition
    # Concat preserves the info better intuitively. Both will work but conceptually concatenate should work better.

    X4 = Concatenate(axis = 1)([X4, X2])
    
    X4 = BatchNormalization()(X4)
    
    X5 = UpSampling2D((2, 2))(X4)
    
#     X5 = Add()([X5, X1])

    X5 = Concatenate(axis = 1)([X5, X1])
    
    decoded = Conv2D(filters = 1, kernel_size = (3, 3), strides = 1, activation = "sigmoid", padding = "same")(X5)
    
    AE = Model(input_image, decoded)
    
    return AE
    


def load_imgs_to_array(path):
    
    """
    This function gets a path where the images are and then crops a 300 x 300 box out of them.
    It returns a np array x = (n_images, width, height, n_channels)   
    """
    
    path_X = path # REMINDER: "/content/drive/My Drive/flickr"

# (X_train, _), (X_test, _) = mnist.load_data()

    file_list = glob.glob(path_X + "/*.jpg")
    file_list.sort()
    x_list = []

    left = 0 # Measured from top left corner. Distances from that corner!
    top = 0
    right = 300
    bottom = 300

    for file in file_list:
        a = Image.open(file)
        b = a.crop((left, top, right, bottom))
        b = np.array(b)
        x_list.append(b)
    
    X = np.array(x_list)
    
    return X
    
def plot_some_pics(n, X):
    plt.figure(figsize = (15,2))

    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    return None
    
    
def Big_UNET(width, height, n_channels):
    """
    This function returns a keras model instance. Compile and fit is done outside.
    """
    
    input_image = Input(shape = (width, height, n_channels)) # Change later.
    
    X1 = Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu", padding = "same", name = "C1")(input_image)
    
    X2 = Conv2D(filters = 128, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C2")(X1)
    
#     X2_N = BatchNormalization(name = "BN2")(X2)
    
    X3 = Conv2D(filters = 256, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C3")(X2)
    
    X4 = Conv2D(filters = 256, kernel_size = (5, 5), strides = 2, activation = "relu", padding = "same", name = "C4")(X3)
    
#     X4_N = BatchNormalization(name = "BN4")(X4)
    
    X5 = Conv2D(filters = 512, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C5")(X4)
    
    encoded = Conv2D(filters = 512, kernel_size = (1, 1), strides = 1, activation = "relu", padding = "same", name = "Encoded")(X5)
    
    # Decoding part
    
    X5T = Conv2DTranspose(filters = 512, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C5T")(encoded)
    
    X40 = ZeroPadding2D(((0, 1), (0, 1)))(X4)
    
    X5T = Concatenate(axis = -1)([X5T, X40])
    
#     X5T = BatchNormalization(name = "BN5T")(X5T)
        
    X4T = Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C4T")(X5T)
    
#     X30 = ZeroPadding2D(((2, 1), (2, 1)))(X3)

    X4T = Cropping2D(((2, 1), (2, 1)))(X4T)
    
    X4T = Concatenate(axis = -1)([X4T, X3])
    
    X3T = Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C3T")(X4T)
    
#     X20 = ZeroPadding2D(((3, 3), (3, 3)))(X2)
    
    X3T = Concatenate(axis = -1)([X3T, X2])
    
#     X3T = BatchNormalization(name = "BN3T")(X3T)
    
    X2T = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C2T")(X3T)
      
#     X1T = BatchNormalization(name = "BN1T")(X1T)
    
    X0T = Conv2D(filters = 3, kernel_size = (3, 3), strides = 1, activation = "sigmoid", padding = "same", name = "C0T")(X2T)
      
    Big_model = Model(input_image, X0T)
    
    return Big_model
    
    
def load_imgs_blurred(path):
    
    """
    This function gets a path where the images are and then crops a 300 x 300 box out of them.
    It returns a np array x = (n_images, width, height, n_channels)   
    """
    
    path_X = path # REMINDER: "/content/drive/My Drive/flickr"

# (X_train, _), (X_test, _) = mnist.load_data()

    file_list = glob.glob(path_X + "/*.jpg")
    file_list.sort()
    x_list = []

    left = 0 # Measured from top left corner. Distances from that corner!
    top = 0
    right = 300
    bottom = 300

    for file in file_list:
        a = Image.open(file)
        b = a.crop((left, top, right, bottom))
        c = b.filter(ImageFilter.GaussianBlur(radius = 20)) # Filter before array!!!
        c = np.array(c)
        x_list.append(c)
    
    X = np.array(x_list)
    
    return X
    
    
def load_imgs_low_res(path):
    
    """
    This function gets a path where the images are and then crops a 300 x 300 box out of them.
    It returns a np array x = (n_images, width, height, n_channels)   
    """
    
    path_X = path # REMINDER: "/content/drive/My Drive/flickr"

# (X_train, _), (X_test, _) = mnist.load_data()

    file_list = glob.glob(path_X + "/*.jpg")
    file_list.sort()
    x_list = []

    left = 0 # Measured from top left corner. Distances from that corner!
    top = 0
    right = 300
    bottom = 300

    for file in file_list:
        a = Image.open(file)
        b = a.crop((left, top, right, bottom))
        c = b.resize((150, 150), resample = Image.BICUBIC) # Downsample with bicubic.
        c = c.resize((300, 300), resample = Image.NEAREST) # Upsample with bilinear, to "spoil" the image a bit. I want to have aliasing!
        c = np.array(c)
        x_list.append(c)
    
    X = np.array(x_list)
    
    return X
    
def PSNR(y_true, y_pred):
    """
    This function evaluates my custom metric, the Peak Signal to Noise Ratio. I remember this from SED or SEP or whatever. The aerodynamic man.
    y_pred is the output of the model and y_true is my label (here, my true image), that I supply to the .fit method. The docs are very confusing.
    """

    max_px = 1.0
    PSNR_value = (20 * K.log(max_px)/K.mean(K.square(y_pred - y_true), axis = -1))
    
    return PSNR_value
    
def split_data(X):
    """
    This function takes a dataset (np array with images) and splits in train, test and validation
    """
    
    X_train = X[:400, :, :] # Remember, this takes from 0 to 799, 800 is taken in the next one
    X_val = X[400:500, :, :] # Math notation = [a, b)
    X_test = X[500:600, :, :]
    
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_val = X_val.astype("float32")

    X_train = X_train/255
    X_test = X_test/255
    X_val = X_val/255

    
    return X_train, X_val, X_test
    
    
    
def Huge_UNET(width, height, n_channels):
    """
    This function returns a keras model instance. Compile and fit is done outside.
    """
    
    input_image = Input(shape = (width, height, n_channels)) # Change later.
    
    X1 = Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu", padding = "same", name = "C1")(input_image)
    
    X2 = Conv2D(filters = 128, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C2")(X1)
    
#     X2_N = BatchNormalization(name = "BN2")(X2)
    
    X3 = Conv2D(filters = 256, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C3")(X2)
    
    X4 = Conv2D(filters = 256, kernel_size = (5, 5), strides = 2, activation = "relu", padding = "same", name = "C4")(X3)
    
#     X4_N = BatchNormalization(name = "BN4")(X4)
    
    X5 = Conv2D(filters = 512, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C5")(X4)
    
    encoded = Conv2D(filters = 1024, kernel_size = (1, 1), strides = 1, activation = "relu", padding = "same", name = "Encoded")(X5)
    
    # Decoding part
    
    X5T = Conv2DTranspose(filters = 1024, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C5T")(encoded)
    
    #X40 = ZeroPadding2D(((0, 1), (0, 1)))(X4)
    
    X5T = Concatenate(axis = -1)([X5T, X4])
    
#     X5T = BatchNormalization(name = "BN5T")(X5T)
        
    X4T = Conv2DTranspose(filters = 512, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C4T")(X5T)
    
    X30 = ZeroPadding2D(((0, 1), (0, 1)))(X3)

    #X4T = Cropping2D(((2, 1), (2, 1)))(X4T)
    
    X4T = Concatenate(axis = -1)([X4T, X30])
    
    X3T = Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C3T")(X4T)
    
    X20 = ZeroPadding2D(((0, 2), (0, 2)))(X2)
    
    X3T = Concatenate(axis = -1)([X3T, X20])
    
    X3T = Cropping2D(((1, 1), (1, 1)))(X3T)
    
#     X3T = BatchNormalization(name = "BN3T")(X3T)
    
    X2T = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = 2, activation = "relu", padding = "same", name = "C2T")(X3T)
      
#     X1T = BatchNormalization(name = "BN1T")(X1T)
    
    X0T = Conv2D(filters = 3, kernel_size = (3, 3), strides = 1, activation = "sigmoid", padding = "same", name = "C0T")(X2T)
      
    Big_model = tf.keras.Model(input_image, X0T)
    
    return Big_model