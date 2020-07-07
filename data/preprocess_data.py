# Importing libraries.
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
import datetime
import numpy as np
import random
import re
import pickle
import plotly.graph_objects as go
from contextlib import redirect_stdout
from keras.preprocessing.image import ImageDataGenerator
from itertools import chain

DATA_DIR = ""
SLICES = [120] # [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]
TRAIN = 0.7
TEST = 0.15
VAL = 0.15
CUT = True # False
RESIZE = (256, 256) # (128, 128)
N_ROTATIONS_TRAIN = 3 # 0
RANGE_ROTATIONS_TRAIN = 20

# Defining function to split images into train and test set. Optinally performing augmentation.
def preprocess_imgs(data_dir, slices, train, val, test, cut = False, resize = False, n_rotations_train = 0, range_rotations_train = 20):
    ## Defining nested function to rotate images.
    def rotate_train_image_pair(train_t1_img, train_t2_img, n_rotations, rotation_range):
        ### Initializing ImageDataGenerator.
        datagen = ImageDataGenerator(rotation_range = range_rotations_train)
        ### Transforming train image to 4D.
        train_t1_img = np.expand_dims(train_t1_img, 0)
        train_t1_img = np.expand_dims(train_t1_img, 3)
        train_t2_img = np.expand_dims(train_t2_img, 0)
        train_t2_img = np.expand_dims(train_t2_img, 3)
        ### Initializing output and number of performed rotations.
        original_t1_t2 = np.concatenate((train_t1_img, train_t2_img), axis = 3)
        output_batch = original_t1_t2
        rotations_done = 0
        ### Performing rotations. Original images are also included in output batch.
        for rotation_batch in datagen.flow(original_t1_t2, batch_size = original_t1_t2.shape[0]):
            if rotations_done >= n_rotations_train:
                break
            output_batch = np.concatenate((output_batch, rotation_batch), axis = 0)
            rotations_done = rotations_done + 1
        ### Returning t1 and t2 batch.
        return output_batch[:,:,:,0], output_batch[:,:,:,1]
    ## Extracting paths for all T1 and T2 images and given slices.
    t1_paths = [glob(data_dir + "/original/subject_*_T1_slice_" + str(slice_elem) + ".png") for slice_elem in SLICES]
    t1_paths = list(chain.from_iterable(t1_paths))
    t2_paths = [glob(data_dir + "/original/subject_*_T2_slice_" + str(slice_elem) + ".png") for slice_elem in SLICES]
    t2_paths = list(chain.from_iterable(t2_paths))
    ## Calculating number of subjects per dataset.
    n_all = len(t1_paths)
    n_train = int(n_all * train)
    n_val = int(n_all * val)
    n_test = int(n_all * test)
    ## Randomly generating indices for training, validation and test set.
    train_indices = random.sample(range(0, n_all), n_train)
    remaining_indices = list(set((range(0, n_all))) - set(train_indices))
    val_indices = random.sample(remaining_indices, n_val)
    test_indices = list(set((remaining_indices)) - set(val_indices))
    # Processing and saving training and test images.
    for dataset in ("train", "val", "test"):
        if dataset == "train":
            os.makedirs(data_dir + "/" + dataset + "/non_rotated/t1")
            os.makedirs(data_dir + "/" + dataset + "/non_rotated/t2")
            if n_rotations_train > 0:
                os.makedirs(data_dir + "/" + dataset + "/rotated/t1")
                os.makedirs(data_dir + "/" + dataset + "/rotated/t2")
        else:
            os.makedirs(data_dir + "/" + dataset + "/t1")
            os.makedirs(data_dir + "/" + dataset + "/t2")
        ## Extracting relevant indices.
        indices = train_indices if dataset == "train" else val_indices if dataset == "val" else test_indices
        ## Performing cutting and resizing.
        for idx in indices:
            ## Reading image as array. Normalizing pixel values to range 0-1.
            t1_img = mpimage.imread(t1_paths[idx]).astype(np.float32)
            t2_img = mpimage.imread(t2_paths[idx]).astype(np.float32)
            ## Cutting rows/columns from image if specified.
            if cut is True:
                t1_img = np.delete(t1_img, (range(15), range(296 - 15, 296)), axis = 0)
                t1_img = np.delete(t1_img, (range(15), range(245 - 15, 245)), axis = 1)
                t2_img = np.delete(t2_img, (range(15), range(296 - 15, 296)), axis = 0)
                t2_img = np.delete(t2_img, (range(15), range(245 - 15, 245)), axis = 1)
            ## Resizing images.
            if resize is not False:
                t1_img = np.array(Image.fromarray(t1_img).resize(size = resize))
                t2_img = np.array(Image.fromarray(t2_img).resize(size = resize))
            ## Optionally rotating train images. Saving.
            if n_rotations_train > 0:
                if dataset == "train":
                    np.save(data_dir + "/train/non_rotated/t1/" + os.path.basename(t1_paths[idx])[:-4] + ".npy", t1_img, allow_pickle = False)
                    np.save(data_dir + "/train/non_rotated/t2/" + os.path.basename(t2_paths[idx])[:-4] + ".npy", t2_img, allow_pickle = False)
                    t1_batch, t2_batch = rotate_train_image_pair(t1_img, t2_img, n_rotations_train, range_rotations_train)
                    iteration = 0
                    for rotated_batch in [t1_batch, t2_batch]:
                        for img_idx in range(rotated_batch.shape[0]):
                            file_ending = "_original" if img_idx == 0 else "_rotated" + str(img_idx) #if img_idx > 0
                            folder_name = "t1/" if iteration == 0 else "t2/" #if iteration > 0
                            paths = t1_paths if iteration == 0 else t2_paths #if iteration > 0
                            np.save(data_dir + "/train/rotated/" + folder_name + os.path.basename(paths[idx])[:-4] + file_ending + ".npy", rotated_batch[img_idx,:,:], allow_pickle = False)
                        iteration = iteration + 1
                else:
                    np.save(data_dir + "/" + dataset + "/t1/" + os.path.basename(t1_paths[idx])[:-4] + ".npy", t1_img, allow_pickle = False)
                    np.save(data_dir + "/" + dataset + "/t2/" + os.path.basename(t2_paths[idx])[:-4] + ".npy", t2_img, allow_pickle = False)
            else:
                if dataset == "train":
                    np.save(data_dir + "/train/non_rotated/t1/" + os.path.basename(t1_paths[idx])[:-4] + ".npy", t1_img, allow_pickle = False)
                    np.save(data_dir + "/train/non_rotated/t2/" + os.path.basename(t2_paths[idx])[:-4] + ".npy", t2_img, allow_pickle = False)
                else:
                    np.save(data_dir + "/" + dataset + "/t1/" + os.path.basename(t1_paths[idx])[:-4] + ".npy", t1_img, allow_pickle = False)
                    np.save(data_dir + "/" + dataset + "/t2/" + os.path.basename(t2_paths[idx])[:-4] + ".npy", t2_img, allow_pickle = False)

preprocess_imgs(DATA_DIR, SLICES, TRAIN, VAL, TEST, CUT, RESIZE, N_ROTATIONS_TRAIN, RANGE_ROTATIONS_TRAIN)
