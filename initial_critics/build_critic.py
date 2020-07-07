from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

IMG_SHAPE = (256, 256, 1) # (128, 128, 1)
CRTIC_LOSS = "binary_crossentropy" # "mse", "kullback_leibler_divergence"
RES_DIR = "../training/initial_critics"

CRITIC_OPTIMIZER = Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999)
BATCHNORM_MOMENTUM = 0.8
DROPOUT_RATE = 0.25
LEAKY_RELU_ALPHA = 0.2
INIT_WEIGHTS = RandomNormal(0., 0.02)
BIAS = True

CRITC_LAYER_STACK = [
    Conv2D(32,  kernel_size = 3, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
    Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(64,  kernel_size = 3, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
    BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(128, kernel_size = 3, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
    BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
    Conv2D(256, kernel_size = 3, strides = 2, padding = "same", kernel_initializer = INIT_WEIGHTS, use_bias = BIAS),
    BatchNormalization(momentum = BATCHNORM_MOMENTUM), Dropout(DROPOUT_RATE), LeakyReLU(LEAKY_RELU_ALPHA),
    Flatten(),
    Dense(1),
    Activation('sigmoid')]

def build_critic(layer_stack, res_dir, img_shape, critic_loss, critic_optimzer):
    if critic_loss == "binary_crossentropy":
        name = "/bincrossentropy_critic_initial_"
    if critic_loss == "mse":
        name = "/mse_critic_initial_"
    if critic_loss == "kullback_leibler_divergence":
        name = "/kl_critic_initial_"
    unknown_image = Input(shape=img_shape)
    critic_output = unknown_image
    for layer in layer_stack:
        critic_output = layer(critic_output)
    critic = Model(inputs=unknown_image, outputs=critic_output)
    critic.compile(loss=critic_loss, optimizer=critic_optimzer)
    critic.save(res_dir + name + str(img_shape[0]) + ".h5")

build_critic(CRITC_LAYER_STACK, RES_DIR, IMG_SHAPE, CRTIC_LOSS, CRITIC_OPTIMIZER)
