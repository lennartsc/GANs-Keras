# Importing libraries.
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
import datetime
import numpy as np
import random
import json
import re
import pickle
import plotly.graph_objects as go
import warnings
from contextlib import redirect_stdout
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, model_from_json
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints
import sys
import evaluation_f
warnings.filterwarnings("ignore")

########################################################################################################################

# Defining GPU to run training on.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################################################################################

# Setting all hyperparameters.

IMG_SHAPE = (256, 256, 1) # (128, 128, 1)
MODEL_DIR = "../dcgan/256_3"
MODEL_TYPE = "dcgan" # "pix2pix2"
MODEL_EPOCH = 2300
RES_DIR = "dcgan__256_3_epoch2300"
Z = (100, "gaussian")
RESCALE_PIXEL_VALUES = True
BATCH_SIZE = 30
D_DIR = "../data/" + str(IMG_SHAPE[0])

########################################################################################################################

# Defining hyperparameters to train critic.

CRITIC_FILE = "../initial_critics/bincrossentropy_critic_initial_256.h5"
CRITIC_TRAIN_EPOCHS = 100
CRITIC_LOSS_MEAN = False
INIT_WEIGHTS = RandomNormal(0., 0.02)
BIAS = True
BATCHNORM_MOMENTUM = 0.8
DROPOUT_RATE = 0.25
LEAKY_RELU_ALPHA = 0.2

########################################################################################################################

# Initializing evaluation framework.

os.mkdir(RES_DIR)
eval_framework = evaluation_f.Evaluation(
    data_dir = D_DIR,
    model_dir = MODEL_DIR,
    model_type = MODEL_TYPE,
    model_epoch = MODEL_EPOCH,
    results_dir = RES_DIR,
    img_shape = IMG_SHAPE,
    z = Z,
    rescale = RESCALE_PIXEL_VALUES
)

########################################################################################################################

# Evaluating.

eval_framework.evaluate(
    batch_size = BATCH_SIZE,
    critic_file = CRITIC_FILE,
    critic_train_epochs = CRITIC_TRAIN_EPOCHS,
    critic_loss_mean = CRITIC_LOSS_MEAN)

########################################################################################################################
