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
from keras.models import Model, model_from_json, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints

class Evaluation():

    def __init__(self, data_dir, model_dir, model_type, model_epoch, results_dir, img_shape, z, rescale):

        self.data_dir = data_dir
        self.train_data_dir = data_dir + "/train/non_rotated"
        self.model_dir = model_dir
        self.model_type = model_type
        self.model_epoch = model_epoch
        self.results_dir = results_dir
        self.img_shape = img_shape
        self.noise_length = z[0]
        self.noise_dist = z[1]
        self.rescale_imgs = rescale
        self.generator = self.load_generator()

    def load_generator(self):
        with open(self.model_dir + "/architecture/generator_architecture.json", 'r') as file:
            generator = file.read()
            generator = model_from_json(generator)
        if self.model_type == "hybrid":
            generator.load_weights(self.model_dir + "/noise1.0/weights/epoch" + str(self.model_epoch) + "_generator_weights.h5")
        else:
            generator.load_weights(self.model_dir + "/weights/epoch" + str(self.model_epoch) + "_generator_weights.h5")
        return generator

    def rescale_pixel_values(self, arr):
        for img in range(arr.shape[0]):
            # Rescaling to [0,1].
            arr[img, :, :] = arr[img, :, :] - np.amin(
                arr[img, :, :])
            if np.amax(arr[img, :, :]) < 1:
                arr[img, :, :] = arr[img, :, :] * (
                        1 / np.amax(arr[img, :, :]))
            elif np.amax(arr[img, :, :]) > 1:
                arr[img, :, :] = arr[img, :, :] / np.amax(
                    arr[img, :, :])
            # Rescaling to [-1,1].
            arr[img, :, :] = (arr[img, :, :] * 2) - 1
            # Returning processed array.
            return arr

    def load_data(self, dataset, batch_size, one_batch = True):
        # data is always shuffled in another way when executing the method.
        data_dict = {}
        if dataset == "train":
            all_t1_paths = glob(self.train_data_dir + "/t1/*")
            all_t2_paths = glob(self.train_data_dir + "/t2/*")
        else:
            all_t1_paths = glob(self.data_dir + "/" + dataset + "/t1/*")
            all_t2_paths = glob(self.data_dir + "/" + dataset + "/t2/*")
        if one_batch:
            indices = random.sample(range(0, len(all_t1_paths)), batch_size)
        else:
            n_all = len(all_t1_paths)
            n_return = int(n_all//batch_size)*batch_size
            indices = random.sample(range(0, len(all_t1_paths)), n_return)
        for t1 in [True, False]:
            if t1:
                paths = all_t1_paths
                type = "t1"
            else:
                paths = all_t2_paths
                type = "t2"
            imgs = []
            for img_idx in indices:
                img = np.load(paths[img_idx])
                if self.rescale_imgs:
                    # Rescaling to [0,1].
                    img = img - np.amin(img)
                    if np.amax(img) < 1:
                        img = img * (1 / np.amax(img))
                    elif np.amax(img) > 1:
                        img = img / np.amax(img)
                    # Rescaling to [-1,1].
                    img = (img * 2) - 1
                imgs.append(img)
            ## Adding width dimension to images. Final returned shape is (batch_size, output_shape, output_shape, 1)
            imgs = np.expand_dims(imgs, 3)
            data_dict[type] = np.array(imgs)
        return data_dict

    def save_samples(self, arr, generated, n_samples):
        if generated:
            folder_name = "/samples_generated"
        else:
            folder_name = "/samples_real"
        # Creating directory to store test samples in.
        if not os.path.exists(self.results_dir + folder_name):
            os.makedirs(self.results_dir + folder_name)
            if generated:
                n_fakes = 0
        else:
            if generated:
                n_fakes = len(glob(self.results_dir + folder_name + "/*png"))
        # Saving each image.
        for img_idx in range(n_samples):
            # Reducing dimensions for plot (needs 2D).
            img = arr[img_idx, :, :, 0]
            if generated and n_fakes > 0:
                save_index = n_fakes + img_idx + 1
            else:
                save_index = img_idx + 1
            # Saving image as png.
            plt.imsave(self.results_dir + folder_name + "/img" + str(save_index) + ".png", img, cmap="gray", dpi=800)
            # Saving image as numpy object.
            np.save(self.results_dir + folder_name + "/img" + str(save_index) + ".npy", img, allow_pickle=False)

    def add_noise(self, arr, noise_weight):
        n, row, col, ch = arr.shape
        if self.noise_dist == "gaussian":
            noise = np.random.normal(0, 1, (n, row, col, ch))
        elif self.noise_dist == "uniform":
            noise = np.random.uniform(0, 1, (n, row, col, ch))
        noise = noise.reshape(n, row, col, ch)
        arr = (1 - noise_weight) * arr + noise_weight * noise
        return arr

    def create_plot(self):
        critic_train_dict = pickle.load(open(self.results_dir + "/critic_training/critic_training.pkl", 'rb'))
        fig = make_subplots(rows = 1, cols = 1)
        fig.add_trace(go.Scatter(x=critic_train_dict["epoch"], y=critic_train_dict["total loss"], name='Total loss', line=dict(color="black")))
        fig.add_trace(go.Scatter(x=critic_train_dict["epoch"], y=critic_train_dict["target loss"], name='Target loss', line=dict(color="red", dash = "dot")))
        fig.add_trace(go.Scatter(x=critic_train_dict["epoch"], y=critic_train_dict["fake loss"], name='Fake loss', line=dict(color="green", dash = "dot")))
        fig.update_layout(xaxis_title="Epoch")
        fig.update_yaxes(title_text="Losses")
        plot(fig, filename=self.results_dir + "/critic_training/critic_training.html", auto_open=False)

    def evaluate(self, batch_size, critic_file, critic_train_epochs, critic_loss_mean):

        # 1 Initializing dictionary with test losses.
        test_losses_dict = {"target loss": [], "fake loss": [], "total loss": []}

        # 2 Training independent critic from scratch.
        critic = load_model(critic_file)
        critic_train_dict = {"epoch": [], "target loss": [], "fake loss": [], "total loss": []}

        for critic_training_epoch in range(critic_train_epochs):
            train_batch = self.load_data("train", batch_size, True)
            train_target_batch = train_batch["t1"]

            if self.model_type == "dcgan":
                if self.noise_dist == "gaussian":
                    noise = np.random.normal(0, 1, (batch_size, 1, 1, self.noise_length))
                elif self.noise_dist == "uniform":
                    noise = np.random.uniform(0, 1, (batch_size, 1, 1, self.noise_length))
                fake_batch = self.generator.predict(noise)
            elif self.model_type == "pix2pix":
                train_input_batch = train_batch["t2"]
                fake_batch = self.generator.predict(train_input_batch)
            else:
                train_input_batch = train_batch["t2"]
                train_input_batch = self.add_noise(train_input_batch, 1.0)
                fake_batch = self.generator.predict(train_input_batch)
            if self.rescale_imgs:
                fake_batch = self.rescale_pixel_values(fake_batch)

            critic_loss_target = critic.train_on_batch(train_target_batch, np.ones((batch_size, 1)))
            critic_loss_fake = critic.train_on_batch(fake_batch, np.zeros((batch_size, 1)))

            if critic_loss_mean is False:
                critic_loss_total = critic_loss_target + critic_loss_fake
            else:
                critic_loss_total = (critic_loss_target + critic_loss_fake) / 2
            critic_train_dict["epoch"].append(critic_training_epoch)
            critic_train_dict["target loss"].append(critic_loss_target)
            critic_train_dict["fake loss"].append(critic_loss_fake)
            critic_train_dict["total loss"].append(critic_loss_total)

        if not os.path.exists(self.results_dir + "/critic_training"):
            os.makedirs(self.results_dir + "/critic_training")
        pickle.dump(critic_train_dict, open(self.results_dir + "/critic_training/critic_training.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
        self.create_plot()

        # Evaluating generator.
        test_all = self.load_data("test", batch_size, False)
        test_target_all = test_all["t1"]
        self.save_samples(test_target_all, False, 100)
        test_input_all = test_all["t2"]
        n_test_all = test_target_all.shape[0]
        n_batches = int(n_test_all / batch_size)

        test_loss_target = 0
        for batch in range(n_batches):
            test_target_batch = test_target_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
            test_loss_target = test_loss_target + critic.test_on_batch(test_target_batch, np.zeros((batch_size, 1)))
        test_loss_target = test_loss_target / n_batches

        test_loss_fake = 0
        n_fakes_saved = 0
        for batch in range(n_batches):
            if self.model_type == "dcgan":
                if self.noise_dist == "gaussian":
                    noise = np.random.normal(0, 1, (batch_size, 1, 1, self.noise_length))
                elif self.noise_dist == "uniform":
                    noise = np.random.uniform(0, 1, (batch_size, 1, 1, self.noise_length))
                fake_batch = self.generator.predict(noise)
            elif self.model_type == "pix2pix":
                test_input_batch = test_input_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                fake_batch = self.generator.predict(test_input_batch)
            elif self.model_type == "hybrid":
                test_input_batch = test_input_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                test_input_batch = self.add_noise(test_input_batch, 1.0)
                fake_batch = self.generator.predict(test_input_batch)
            if self.rescale_imgs:
                fake_batch = self.rescale_pixel_values(fake_batch)
            if n_fakes_saved <= 80:
                self.save_samples(fake_batch, True, 20)
            test_loss_fake = test_loss_fake + critic.test_on_batch(fake_batch, np.ones((batch_size, 1)))
        test_loss_fake = test_loss_fake / n_batches

        if critic_loss_mean is False:
            test_loss_total = test_loss_target + test_loss_fake
        else:
            test_loss_total = (test_loss_target + test_loss_fake) / 2

        # Saving test losses.
        test_losses_dict["target loss"].append(test_loss_target)
        test_losses_dict["fake loss"].append(test_loss_fake)
        test_losses_dict["total loss"].append(test_loss_total)
        with open(self.results_dir + "/test_losses.txt", "w") as file:
            file.write(json.dumps(test_losses_dict))

