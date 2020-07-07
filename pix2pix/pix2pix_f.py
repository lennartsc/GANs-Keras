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
import psutil
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
class Pix2pix():

    def __init__(self, data_dir, results_dir, img_shape, disc_dict, gen_dict, u_net = True, rescale = False, rotated_train = True):
        ## Defining directories, rescale setting and image shape.
        self.data_dir = data_dir
        if rotated_train:
            self.train_data_dir = data_dir + "/train/rotated"
        else:
            self.train_data_dir = data_dir + "/train/non_rotated"
        self.results_dir = results_dir
        self.rescale_imgs = rescale
        self.image_shape = img_shape
        ## Building discriminator.
        self.discriminator = self.build_discriminator(disc_dict["layer_stack"])
        self.discriminator.compile(loss = disc_dict["loss"], optimizer = disc_dict["optimizer"])
        ## Building generator. Inmplementing it as a combined model of generator and non-trainable discriminator.
        self.generator = self.build_generator(gen_dict["down_layer_stack"], gen_dict["up_layer_stack"], concatenate = u_net)
        input_img = Input(shape = self.image_shape)
        fake_img = self.generator(input_img)
        self.discriminator.trainable = False
        patch_gan_fake = self.discriminator(inputs = [input_img, fake_img])
        self.combined = Model(inputs = input_img, outputs = [patch_gan_fake, fake_img])
        self.gen_loss_lambda = gen_dict["loss"][2]
        self.combined.compile(loss =  gen_dict["loss"][:2], loss_weights = [1, self.gen_loss_lambda], optimizer = gen_dict["optimizer"])
        # Saving model architectures as txt files.
        if not os.path.exists(self.results_dir + "/architecture"):
            os.makedirs(self.results_dir + "/architecture")
        with open(self.results_dir + "/architecture/discriminator_architecture.txt", 'w') as f:
            with redirect_stdout(f):
                self.discriminator.summary()
        with open(self.results_dir + "/architecture/generator_architecture.txt", 'w') as f:
            with redirect_stdout(f):
                self.generator.summary()
        with open(self.results_dir + "/architecture/generator_architecture.json", "w") as json_file:
            json_file.write(self.generator.to_json())

    def build_discriminator(self, layer_stack):
        ## Instantiating.
        input_img = Input(shape = self.image_shape)
        unknown_img = Input(shape = self.image_shape) # unknown_img either target image or fake image.
        patch_gan = Concatenate(axis=-1)([input_img, unknown_img])
        ## Downsampling through the model.
        for layer in layer_stack:
            patch_gan = layer(patch_gan)
        ## Returning discriminator.
        return Model(inputs = [input_img, unknown_img], outputs = patch_gan)

    def build_generator(self, down_layer_stack, up_layer_stack, concatenate):
        ## Instantiating.
        input_img = Input(shape = self.image_shape)
        fake_img = input_img
        ## Downsampling through the model. Saving skip connection after every LeakyReLU.
        skips = []
        for down_layer in down_layer_stack:
            fake_img = down_layer(fake_img)
            if isinstance(down_layer, LeakyReLU):
                skips.append(fake_img)
        skips = list(reversed(skips[:-1]))
        ## Upsampling and establishing the skip connections.
        skip_idx = 0
        for up_layer in up_layer_stack:
            fake_img = up_layer(fake_img)
            if concatenate:
                if isinstance(up_layer, LeakyReLU):
                    fake_img = Concatenate()([fake_img, skips[skip_idx]])
                    skip_idx = skip_idx + 1
        ## Returning generator.
        return Model(inputs = input_img, outputs = fake_img)

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

    def save_fake_samples(self, fake_imgs, epoch, n_samples):
        # Creating target folders if they do not exist.
        if not os.path.exists(self.results_dir + "/train_samples"):
            os.makedirs(self.results_dir + "/train_samples")
        # Sampling n_samples images. Saving plots.
        for img_idx in range(n_samples):
            # Reducing dimensions for plot (needs 2D).
            fake_img = fake_imgs[img_idx, :, :, 0]
            # Saving image,
            plt.imsave(self.results_dir + "/train_samples/epoch" + str(epoch) + "_" + str(img_idx+1) + ".png", fake_img, cmap = "gray", dpi = 800)

    def create_plot(self, n_epochs, evaluation_interval):
        ## Critic training.
        n_eval_epochs = n_epochs//evaluation_interval
        eval_epochs = [0]
        for each in range(n_eval_epochs):
            eval_epochs.append(eval_epochs[-1] + evaluation_interval)
        for eval_epoch in eval_epochs:
            critic_train_dict = pickle.load(open(self.results_dir + "/logs/critic_training/epoch" + str(eval_epoch) + ".pkl", 'rb'))
            fig = make_subplots(rows = 1, cols = 1)
            fig.add_trace(go.Scatter(x=critic_train_dict["epoch"], y=critic_train_dict["total loss"], name='Total loss', line=dict(color="black")))
            fig.add_trace(go.Scatter(x=critic_train_dict["epoch"], y=critic_train_dict["target loss"], name='Target loss', line=dict(color="red", dash = "dot")))
            fig.add_trace(go.Scatter(x=critic_train_dict["epoch"], y=critic_train_dict["fake loss"], name='Fake loss', line=dict(color="green", dash = "dot")))
            fig.update_layout(xaxis_title="Epoch")
            fig.update_yaxes(title_text="Losses")
            plot(fig, filename=self.results_dir + "/logs/critic_training/epoch" + str(eval_epoch) + ".html", auto_open=False)
        ## Validation and Disc losses.
        val_losses_dict = pickle.load(open(self.results_dir + "/logs/val_losses.pkl", 'rb'))
        disc_losses_dict = pickle.load(open(self.results_dir + "/logs/disc_losses.pkl", 'rb'))
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=val_losses_dict["epoch"], y=val_losses_dict["total loss"], name='Total val loss', line=dict(color="black")))
        fig.add_trace(go.Scatter(x=disc_losses_dict["epoch"], y=disc_losses_dict["total loss"], name='Total disc loss', line=dict(color="black",dash="dot")))
        fig.add_trace(go.Scatter(x=val_losses_dict["epoch"], y=val_losses_dict["target loss"], name='Target val loss', line=dict(color="red")))
        fig.add_trace(go.Scatter(x=disc_losses_dict["epoch"], y=disc_losses_dict["target loss"], name='Target disc loss', line=dict(color="red", dash="dot")))
        fig.add_trace(go.Scatter(x=val_losses_dict["epoch"], y=val_losses_dict["fake loss"], name='Fake val loss', line=dict(color="green")))
        fig.add_trace(go.Scatter(x=disc_losses_dict["epoch"], y=disc_losses_dict["fake loss"], name='Fake disc loss', line=dict(color="green", dash="dot")))
        fig.update_layout(xaxis_title="Epoch")
        fig.update_yaxes(title_text="Losses")
        plot(fig, filename=self.results_dir + "/logs/val_disc_losses.html", auto_open=False)

    def train(self, n_epochs, batch_size, evaluation_interval, critic_file, critic_train_epochs, n_saved_samples, disc_critic_loss_mean, save_weights):

        # 1 Setting start time of training.
        start_time = datetime.datetime.now()

        # 2 Extracting patch_gan shape.
        patch_gan_shape = self.discriminator.layers[-1].output_shape[1:4]

        # 3 Initializing dictionary with validation losses.
        val_losses_dict = {"epoch": [], "target loss": [], "fake loss": [], "total loss": []}
        disc_losses_dict = {"epoch": [], "target loss": [], "fake loss": [], "total loss": []}

        # 4 Running training.
        for epoch in range(n_epochs + 1):

            # 4.1 Evaluating generator.
            if (epoch % evaluation_interval) == 0:

                print("[Start evaluation epoch: %d/%d]" % (epoch, n_epochs))
                print(psutil.virtual_memory())

                # 4.1.1 Training independent critic from scratch. Generator stays fixed. Logging critic training losses.
                if epoch == 0:
                    critic = load_model(critic_file)
                    critic_intial_weights = critic.get_weights()
                else:
                    critic.set_weights(critic_intial_weights)

                critic_train_dict = {"epoch": [], "target loss": [], "fake loss": [], "total loss": []}

                for critic_training_epoch in range(critic_train_epochs):

                    train_batch = self.load_data("train", batch_size, True)
                    train_target_batch = train_batch["t1"]
                    train_input_batch = train_batch["t2"]
                    fake_batch = self.generator.predict(train_input_batch)
                    if self.rescale_imgs:
                        fake_batch = self.rescale_pixel_values(fake_batch)
                    critic_loss_target = critic.train_on_batch(train_target_batch, np.ones((batch_size, 1)))
                    critic_loss_fake = critic.train_on_batch(fake_batch, np.zeros((batch_size, 1)))
                    if disc_critic_loss_mean is False:
                        critic_loss_total = critic_loss_target + critic_loss_fake
                    else:
                        critic_loss_total = (critic_loss_target + critic_loss_fake) / 2
                    critic_train_dict["epoch"].append(critic_training_epoch)
                    critic_train_dict["target loss"].append(critic_loss_target)
                    critic_train_dict["fake loss"].append(critic_loss_fake)
                    critic_train_dict["total loss"].append(critic_loss_total)

                if not os.path.exists(self.results_dir + "/logs/critic_training"):
                    os.makedirs(self.results_dir + "/logs/critic_training")
                pickle.dump(critic_train_dict, open(self.results_dir + "/logs/critic_training/epoch" + str(epoch) + ".pkl", 'wb'), pickle.HIGHEST_PROTOCOL)

                # 4.1.2 Evaluating fixed generator using critic and validation data.
                val_all = self.load_data("val", batch_size, False)
                val_target_all = val_all["t1"]
                val_input_all = val_all["t2"]
                n_val = val_target_all.shape[0]
                n_batches = int(n_val / batch_size)

                val_loss_target = 0
                for batch in range(n_batches):
                    val_target_batch = val_target_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                    val_loss_target = val_loss_target + critic.test_on_batch(val_target_batch, np.zeros((batch_size, 1)))
                val_loss_target = val_loss_target / n_batches

                val_loss_fake = 0
                for batch in range(n_batches):
                    val_input_batch = val_input_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                    fake_batch = self.generator.predict(val_input_batch)
                    if self.rescale_imgs:
                        fake_batch = self.rescale_pixel_values(fake_batch)
                    val_loss_fake = val_loss_fake + critic.test_on_batch(fake_batch, np.ones((batch_size, 1)))
                val_loss_fake = val_loss_fake / n_batches

                if disc_critic_loss_mean is False:
                    val_loss_total = val_loss_target + val_loss_fake
                else:
                    val_loss_total = (val_loss_target + val_loss_fake) / 2
                val_losses_dict["epoch"].append(epoch)
                val_losses_dict["target loss"].append(val_loss_target)
                val_losses_dict["fake loss"].append(val_loss_fake)
                val_losses_dict["total loss"].append(val_loss_total)

                # 4.1.3 Logging actual discriminator losses for comparison.
                disc_loss_target = 0
                for batch in range(n_batches):
                    val_input_batch = val_input_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                    val_target_batch = val_target_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                    disc_loss_target = disc_loss_target + self.discriminator.test_on_batch([val_input_batch, val_target_batch], np.zeros((batch_size,) + patch_gan_shape))
                disc_loss_target = disc_loss_target / n_batches

                disc_loss_fake = 0
                for batch in range(n_batches):
                    val_input_batch = val_input_all[batch * batch_size:batch * batch_size + batch_size, :, :, ]
                    fake_batch = self.generator.predict(val_input_batch)
                    if self.rescale_imgs:
                        fake_batch = self.rescale_pixel_values(fake_batch)
                    disc_loss_fake = disc_loss_fake + self.discriminator.test_on_batch([val_input_batch, fake_batch], np.ones((batch_size,) + patch_gan_shape))
                disc_loss_fake = disc_loss_fake / n_batches

                if disc_critic_loss_mean is False:
                    disc_loss_total = disc_loss_target + disc_loss_fake
                else:
                    disc_loss_total = (disc_loss_target + disc_loss_fake) / 2
                disc_losses_dict["epoch"].append(epoch)
                disc_losses_dict["target loss"].append(disc_loss_target)
                disc_losses_dict["fake loss"].append(disc_loss_fake)
                disc_losses_dict["total loss"].append(disc_loss_total)

                # 4.1.4 Saving current samples and weights of generator.
                self.save_fake_samples(fake_batch, epoch, n_saved_samples)
                if save_weights is True:
                    if not os.path.exists(self.results_dir + "/weights"):
                        os.makedirs(self.results_dir + "/weights")
                    self.generator.save_weights(
                        self.results_dir + "/weights/epoch" + str(epoch) + "_generator_weights.h5")
                    self.discriminator.save_weights(
                        self.results_dir + "/weights/epoch" + str(epoch) + "_discriminator_weights.h5")

                # 4.1.5 Printing evaluation summary.
                print("[Epoch: %d/%d] [Validation loss: %f] [Time: %s]" % (epoch, n_epochs, val_loss_total, datetime.datetime.now() - start_time))

            # 4.2 Training discriminator and generator.
            train_batch = self.load_data("train", batch_size, True)
            train_target_batch = train_batch["t1"]
            train_input_batch = train_batch["t2"]
            fake_batch = self.generator.predict(train_input_batch)
            if self.rescale_imgs:
                fake_batch = self.rescale_pixel_values(fake_batch)
            self.discriminator.train_on_batch([train_input_batch, train_target_batch], np.ones((batch_size,) + patch_gan_shape))
            self.discriminator.train_on_batch([train_input_batch, fake_batch], np.zeros((batch_size,) + patch_gan_shape))
            self.combined.train_on_batch(train_input_batch, [np.ones((batch_size,) + patch_gan_shape), train_target_batch])

        # 5 Logging and plotting validation and disc losses.
        pickle.dump(val_losses_dict, open(self.results_dir + "/logs/val_losses.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(disc_losses_dict, open(self.results_dir + "/logs/disc_losses.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
        self.create_plot(n_epochs, evaluation_interval)