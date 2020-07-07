import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
import datetime
import numpy as np
import random
import re
import psutil
import pickle
import warnings
import plotly.graph_objects as go
from contextlib import redirect_stdout
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from keras.layers import Input, Dropout, Concatenate
from keras.layers import Dense, Reshape, Flatten
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.initializers import RandomNormal
from keras.models import Model, load_model
from keras.optimizers import Adam
import sys
import pix2pix_f

warnings.filterwarnings("ignore")

########################################################################################################################

# Defining GPU to run training on.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################################################################################

# Creating folder to store results in.

RES_DIR = os.path.splitext(os.path.basename(__file__))[0]
os.mkdir(RES_DIR)

########################################################################################################################

# Defining hyperparameters.

IMG_SHAPE = (256, 256, 1)
CRITIC_FILE = "../initial_critics/bincrossentropy_critic_initial_256.h5"
D_DIR = "../data/" + str(IMG_SHAPE[0])
EPOCHS = 3000
BATCH_SIZE = 30
EVALUATION_INTERVAL = 100
CRITIC_TRAIN_EPOCHS = 200
SAVE_WEIGHTS = True


INIT_WEIGHTS = RandomNormal(0., 0.02)
DISC_OPTIMIZER = Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999)
GEN_OPTIMIZER = Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999)
DISC_LOSS = "binary_crossentropy"
GEN_LOSS = ("binary_crossentropy", "mae", 100)
BIAS = True
BATCHNORM_MOMENTUM = 0.8
DROPOUT_RATE = 0.25
LEAKY_RELU_ALPHA = 0.2
N_SAVED_SAMPLES = 2
DISC_CRITIC_LOSS_MEAN = False
RESCALE_PIXEL_VALUES = True
ROTATED_TRAIN = False
U_NET = True

########################################################################################################################

# Defining layer stacks.

if IMG_SHAPE == (256, 256, 1):
    GEN_DOWN_LAYER_STACK = [
        Conv2D(64, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(128, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(256, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA)]
    GEN_UP_LAYER_STACK = [
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(256, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(128, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(64, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(1, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        Activation('tanh')]
    DISC_LAYER_STACK = [
        Conv2D(64, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        LeakyReLU(),
        Conv2D(128, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(),
        Conv2D(256, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(),
        ZeroPadding2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "valid", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(),
        ZeroPadding2D(),
        Conv2D(1, kernel_size = 4, strides = 1, padding = "valid", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        Activation('tanh')]

if IMG_SHAPE == (128, 128, 1):
    GEN_DOWN_LAYER_STACK = [
        Conv2D(64, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(128, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(256, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        Conv2D(512, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA)]
    GEN_UP_LAYER_STACK = [
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(512, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(256, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(128, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(64, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(LEAKY_RELU_ALPHA),
        UpSampling2D(),
        Conv2D(1, kernel_size = 4, strides = 1, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        Activation('tanh')]
    DISC_LAYER_STACK = [
        Conv2D(64, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        LeakyReLU(),
        Conv2D(128, kernel_size = 4, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(),
        ZeroPadding2D(),
        Conv2D(256, kernel_size = 4, strides = 1, padding = "valid", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        BatchNormalization(momentum = BATCHNORM_MOMENTUM), LeakyReLU(),
        ZeroPadding2D(),
        Conv2D(1, kernel_size = 4, strides = 1, padding = "valid", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
        Activation('tanh')]

########################################################################################################################

# Creating GAN using given hyperparameters and specified architecture.

gan = pix2pix_f.Pix2pix(
    data_dir=D_DIR,
    results_dir=RES_DIR,
    img_shape=IMG_SHAPE,
    disc_dict={
        "layer_stack": DISC_LAYER_STACK,
        "loss": DISC_LOSS,
        "optimizer": DISC_OPTIMIZER
    },
    gen_dict={
        "up_layer_stack": GEN_UP_LAYER_STACK,
        "down_layer_stack": GEN_DOWN_LAYER_STACK,
        "loss": GEN_LOSS,
        "optimizer": GEN_OPTIMIZER
    },
    u_net = U_NET,
    rescale=RESCALE_PIXEL_VALUES,
    rotated_train=ROTATED_TRAIN)

########################################################################################################################

# Training GAN.

gan.train(n_epochs = EPOCHS,
          batch_size = BATCH_SIZE,
          evaluation_interval = EVALUATION_INTERVAL,
          critic_file = CRITIC_FILE,
          critic_train_epochs = CRITIC_TRAIN_EPOCHS,
          n_saved_samples = N_SAVED_SAMPLES,
          disc_critic_loss_mean = DISC_CRITIC_LOSS_MEAN,
          save_weights = SAVE_WEIGHTS)

########################################################################################################################
