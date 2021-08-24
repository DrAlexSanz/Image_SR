import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os, shutil
from distutils.dir_util import copy_tree

def load_pictures(path):
    """
    This function takes the path of a given folder where the pictures are and loads them.
    
    Input: the path, using glob I read all the images
    Output: The X vector. A numpy array with all the pictures. Maybe reading 800 pictures in an array is not the best way.
    """
    
    file_list = glob.glob(path + "/*.png") # Make sure it's the correct directory, otherwise there's no warning!
    file_list.sort()

    X = np.array([plt.imread(f_name) for f_name in file_list])
    
    return X
    
def rename_old(path):
    """
    Someone decided to name the pictures in a retarded way. I will have to rename everything after I load. And then I copy them.
    Input is the path, returns Nothing    
    """
    os.chdir(path)
    
    src_list = glob.glob("*") # Make sure it's the correct directory, otherwise there's no warning!
    src_list.sort()
    for i in range(800):
        src = src_list[i]
        dest = "0" + str(i + 1) + ".png"
        os.rename(src, dest)

    return
    
 

def copy_files(source, parent_dest):

    """
    This function copies the files I have in my drive into a different one in colab. Just so I don't mess everything up without noticing.
    Not the fastest method, but doable.
    """

    os.chdir("/content")
    
    if os.path.isdir("/content/Image-SR") == True:
        shutil.rmtree("/content/Image-SR")
    
    #parent_dest = "/content/Image-SR"
    
    #source = "/content/drive/My Drive/Imge-SR"


    shutil.copytree(source, parent_dest)

    
    
    return
    
    
    
 
def make_val_set(parent_val, val_HR, val_LR):

    """
    This function creates a directory val (with HR and LR subdirectories) and it will copy the last 50 train pictures to use as a validation set.
    
    Input: nothing
    Output: True if the execution was correct.    
    """
        
    if os.path.isdir("/content/drive/MyDrive/Image-SR/val") == True:
        shutil.rmtree("/content/drive/MyDrive/Image-SR/val")
    
    os.mkdir(parent_val)
    #os.mkdir(val_HR)
    #os.mkdir(val_LR)
    
    # Now I move stuff with a for loop. It may not be too quick but there are only 50 images per folder that I want to move

    for i in range(751, 801):
        path_source = "/content/drive/MyDrive/Image-SR/train/HR/0" + str(i) + ".png"
        shutil.move(path_source, val_HR)
        
    for i in range(751, 801):
        path_source = "/content/drive/MyDrive/Image-SR/train/LR/0" + str(i) + ".png"
        shutil.move(path_source, val_LR)
    
    #I don't really want to return anything, just operate.
    return
    
    
def explore_dimensions(path):
    """
    This function explores the dimensions of the dataset and plots the height and width of the train and test pictures
    With this data, I will decide what is the input and output size. Remember the shape in plt.imread is always (height, width)
    
    Input: is the path where the pictures are located
    
    Output: The output is the mean and median of the train and test datasets (train_mean, train_median, test_mean, test_median)
    """
    
    # just a reminder of the structure, I get path as an argument.
    # path_train_HR = "/content/Image-SR/train/HR"
    # path_train_LR = "/content/Image-SR/train/LR"
    
    os.chdir(path)
    
    file_list = glob.glob(path + "/*.png") # Make sure it's the correct directory, otherwise there's no warning!
    file_list.sort()
    
    height = [Image.open(name).size[0] for name in file_list]

    width = [Image.open(name).size[1] for name in file_list]
    
    # Not directly useful I think, calculate just in case I want them in the future
    ratios = list(np.array(height) / np.array(width))

    dim = [Image.open(name).size for name in file_list]
    
    # Now plot the occurrences of height and width
    
    x_h, y_h = np.unique(height, return_counts = True)
    
    x_w, y_w = np.unique(width, return_counts = True)
    
    plt.figure(figsize = (15, 12))
    
    plt.subplot(2, 1, 1)
    plt.title("Height plot")
    plt.scatter(x_h, y_h)
    
    plt.subplot(2, 1, 2)
    plt.title("Width plot")
    plt.scatter(x_w, y_w)
    
    mean_h = int(np.mean(height))
    median_h = np.median(height)
    
    mean_w = int(np.mean(width))
    median_w = np.median(width)
    
    print("The mean of the heights is: ", mean_h)
    print("The median of the heights is: ", median_h)

    print("The mean of the widths is: ", mean_w)
    print("The median of the widths is: ", median_w)
    
    os.chdir("/content/Image-SR")
    
    aux = (mean_h, median_h, mean_w, median_w)
    
    return aux
    
    
def rename(path):
    """
    This function renames the LR images. Someone decided to name them AAAA2x.png
    The strategy is to use rstrip for the last characters and add .png later
    """
    
    #path = "/content/Image-SR/test/LR"
    os.chdir(path)
    
    for filename in os.listdir("."):
        if filename.endswith("x2.png"):
            os.rename(filename, filename.replace("x2.png", ".png"))
            
    return
    
    
def resize_input(height, width, path):
    """
    This function takes the LR images and resizes them to the desired input. For a SR autoencoder the input and output should be the same picture, but the input should have lower resolution.
    Does not return anything.
    I can do it using flow_from_directory and do everything on the fly but that requires doing it more times when I don't want to perform data augmentation.
    I can always not use this function in the future.
    """
    
    # Quick reminder
    # path_test = "/content/Image-SR/test/LR"
    # path_train = "/content/Image-SR/train/LR"
    # path_val = "/content/Image-SR/val/LR"
    
    os.chdir(path)
    
    for fname in os.listdir("."):
        X_LR = Image.open(fname)
        X_LR = X_LR.resize((width, height), Image.BILINEAR)
        X_LR.save(path + "/" + fname, format = "png")
    
    
    
    return