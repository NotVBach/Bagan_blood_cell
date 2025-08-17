"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import os
import csv
from utils import save_image_array

K.set_image_data_format('channels_first')

class BalancingGAN:
    def build_generator(self, latent_size, init_resolution=8):
        resolution = self.resolution
        channels = self.channels

        cnn = Sequential()
        cnn.add(Dense(1024, input_dim=latent_size, activation='relu', use_bias=False))
        cnn.add(Dense(128 * init_resolution * init_resolution, activation='relu', use_bias=False))
        cnn.add(Reshape((128, init_resolution, init_resolution)))
        crt_res = init_resolution

        while crt_res != resolution:
            cnn.add(UpSampling2D(size=(2, 2)))
            if crt_res < resolution/2:
                cnn.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_normal', use_bias=False))
            else:
                cnn.add(Conv2D(128, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_normal', use_bias=False))
            crt_res *= 2
            assert crt_res <= resolution, f"Error: final resolution [{resolution}] must equal i*2^n."

        cnn.add(Conv2D(channels, (2, 2), padding='same', activation='tanh', kernel_initializer='glorot_normal', use_bias=False))
        latent = Input(shape=(latent_size,))
        fake_image_from_latent = cnn(latent)
        self.generator = Model(inputs=latent, outputs=fake_image_from_latent)

    def _build_common_encoder(self, image, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        cnn = Sequential()
        cnn.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2), input_shape=(channels, resolution, resolution), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        while cnn.output_shape[-1] > min_latent_res:
            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))
            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

        cnn.add(Flatten())
        features = cnn(image)
        return features

    def build_reconstructor(self, latent_size, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(channels, resolution, resolution))
        features = self._build_common_encoder(image, min_latent_res)
        latent = Dense(latent_size, activation='linear')(features)
        self.reconstructor = Model(inputs=image, outputs=latent)

    def build_discriminator(self, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(channels, resolution, resolution))
        features = self._build_common_encoder(image, min_latent_res)
        aux = Dense(self.nclasses + 1, activation='softmax', name='auxiliary')(features)
        self.discriminator = Model(inputs=image, outputs=aux)

    def generate_from_latent(self, latent):
        return self.generator(latent)

    def generate(self, c, bg=None):
        latent = self.generate_latent(c, bg)
        return self.generator.predict(latent, verbose=0)

    def generate_latent(self, c, bg=None, n_mix=10):
        res = np.array([np.random.multivariate_normal(self.means[self.label_map[e]], self.covariances[self.label_map[e]]) for e in c])
        return res

    def discriminate(self, image):
        return self.discriminator(image)

    def __init__(self, classes, target_class_id, dratio_mode="uniform", gratio_mode="uniform", adam_lr=0.00005, latent_size=100, res_dir="./res-tmp", image_shape=[3, 32, 32], min_latent_res=8):
        self.gratio_mode = gratio_mode
        self.dratio_mode = dratio_mode
        self.classes = classes
        self.target_class_id = target_class_id
        self.nclasses = len(classes)
        self.label_map = {c: i for i, c in enumerate(classes)}
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[0]
        self.resolution = image_shape[1]
        if self.resolution != image_shape[2]:
            raise ValueError("Only square images supported")
        self.min_latent_res = min_latent_res
        self.adam_lr = adam_lr
        self.adam_beta_1 = 0.5
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        self.build_generator(latent_size, init_resolution=min_latent_res)
        self.generator.compile(optimizer=Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta_1), loss='sparse_categorical_crossentropy')
        latent_gen = Input(shape=(latent_size,))
        self.build_discriminator(min_latent_res=min_latent_res)
        self.discriminator.compile(optimizer=Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta_1), loss='sparse_categorical_crossentropy')
        self.build_reconstructor(latent_size, min_latent_res=min_latent_res)
        self.reconstructor.compile(optimizer=Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta_1), loss='mean_squared_error')

        fake = self.generator(latent_gen)
        self.discriminator.trainable = False
        self.reconstructor.trainable = False
        self.generator.trainable = True
        aux = self.discriminate(fake)
        self.combined = Model(inputs=latent_gen, outputs=aux)
        self.combined.compile(optimizer=Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta_1), loss='sparse_categorical_crossentropy')

        self.discriminator.trainable = False
        self.generator.trainable = True
        self.reconstructor.trainable = True
        img_for_reconstructor = Input(shape=(self.channels, self.resolution, self.resolution))
        img_reconstruct = self.generator(self.reconstructor(img_for_reconstructor))
        self.autoenc_0 = Model(inputs=img_for_reconstructor, outputs=img_reconstruct)
        self.autoenc_0.compile(optimizer=Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta_1), loss='mean_squared_error')

    def _biased_sample_labels(self, samples, target_distribution="uniform"):
        distribution = self.class_uratio
        if target_distribution == "d":
            distribution = self.class_dratio
        elif target_distribution == "g":
            distribution = self.class_gratio
        sampled_labels = np.full(samples, self.classes[0])
        sampled_labels_p = np.random.uniform(0, 1, samples)
        for c in range(self.nclasses):
            mask = np.logical_and((sampled_labels_p > 0), (sampled_labels_p <= distribution[c]))
            sampled_labels[mask] = self.classes[c]
            sampled_labels_p -= distribution[c]
        return sampled_labels

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []
        for image_batch, label_batch in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]
            fake_size = int(np.ceil(crt_batch_size * 1.0 / self.nclasses))
            sampled_labels = self._biased_sample_labels(fake_size, "d")
            latent_gen = self.generate_latent(sampled_labels, bg_train)
            generated_images = self.generator.predict(latent_gen, verbose=0)
            X = np.concatenate((image_batch, generated_images))
            aux_y = np.concatenate((label_batch, np.full(len(sampled_labels), max(self.classes) + 1)), axis=0)
            epoch_disc_loss.append(self.discriminator.train_on_batch(X, aux_y))
            sampled_labels = self._biased_sample_labels(fake_size + crt_batch_size, "g")
            latent_gen = self.generate_latent(sampled_labels, bg_train)
            epoch_gen_loss.append(self.combined.train_on_batch(latent_gen, sampled_labels))
        return np.mean(np.array(epoch_disc_loss), axis=0), np.mean(np.array(epoch_gen_loss), axis=0)

    def _set_class_ratios(self):
        self.class_dratio = np.full(self.nclasses, 0.0)
        target = 1 / self.nclasses
        self.class_uratio = np.full(self.nclasses, target)
        self.class_gratio = np.full(self.nclasses, 0.0)
        for c in range(self.nclasses):
            if self.gratio_mode == "uniform":
                self.class_gratio[c] = target
            elif self.gratio_mode == "rebalance":
                self.class_gratio[c] = 2 * target - self.class_aratio[c]
            else:
                raise ValueError(f"Unknown gmode {self.gratio_mode}")
        for c in range(self.nclasses):
            if self.dratio_mode == "uniform":
                self.class_dratio[c] = target
            elif self.dratio_mode == "rebalance":
                self.class_dratio[c] = 2 * target - self.class_aratio[c]
            else:
                raise ValueError(f"Unknown dmode {self.dratio_mode}")
        if self.gratio_mode == "rebalance":
            self.class_gratio[self.class_gratio < 0] = 0
            self.class_gratio /= sum(self.class_gratio)
        if self.dratio_mode == "rebalance":
            self.class_dratio[self.class_dratio < 0] = 0
            self.class_dratio /= sum(self.class_dratio)

    def init_autoenc(self, bg_train, gen_fname=None, rec_fname=None):
        generator_fname = gen_fname if gen_fname else f"{self.res_dir}/{self.target_class_id}_generator.weights.h5"
        reconstructor_fname = rec_fname if rec_fname else f"{self.res_dir}/{self.target_class_id}_reconstructor.weights.h5"
        multivariate_prelearnt = False

        if os.path.exists(generator_fname) and os.path.exists(reconstructor_fname):
            print(f"BAGAN: loading autoencoder: {generator_fname}, {reconstructor_fname}")
            self.generator.load_weights(generator_fname)
            self.reconstructor.load_weights(reconstructor_fname)
            if os.path.exists(f"{self.res_dir}/{self.target_class_id}_means.npy") and os.path.exists(f"{self.res_dir}/{self.target_class_id}_covariances.npy"):
                multivariate_prelearnt = True
                cfname = f"{self.res_dir}/{self.target_class_id}_covariances.npy"
                mfname = f"{self.res_dir}/{self.target_class_id}_means.npy"
                print(f"BAGAN: loading multivariate: {cfname}, {mfname}")
                self.covariances = np.load(cfname)
                self.means = np.load(mfname)
        else:
            print("BAGAN: training autoencoder")
            autoenc_train_loss = []
            for e in range(self.autoenc_epochs):
                print(f'Autoencoder train epoch: {e+1}/{self.autoenc_epochs}')
                autoenc_train_loss_crt = []
                for image_batch, _ in bg_train.next_batch():
                    autoenc_train_loss_crt.append(self.autoenc_0.train_on_batch(image_batch, image_batch))
                autoenc_train_loss.append(np.mean(np.array(autoenc_train_loss_crt), axis=0))
            autoenc_loss_fname = f"{self.res_dir}/{self.target_class_id}_autoencoder.csv"
            with open(autoenc_loss_fname, 'w') as csvfile:
                for item in autoenc_train_loss:
                    csvfile.write(f"{item}\n")
            self.generator.save_weights(generator_fname)
            self.reconstructor.save_weights(reconstructor_fname)

        layers_r = self.reconstructor.layers
        layers_d = self.discriminator.layers
        for l in range(1, len(layers_r)-1):
            layers_d[l].set_weights(layers_r[l].get_weights())

        if not multivariate_prelearnt:
            print("BAGAN: computing multivariate")
            self.covariances = []
            self.means = []
            for c in self.classes:
                imgs = bg_train.get_samples_for_class(c, len(bg_train.per_class_ids[c]))
                latent = self.reconstructor.predict(imgs, verbose=0)
                self.covariances.append(np.cov(np.transpose(latent)))
                self.means.append(np.mean(latent, axis=0))
            self.covariances = np.array(self.covariances)
            self.means = np.array(self.means)
            cfname = f"{self.res_dir}/{self.target_class_id}_covariances.npy"
            mfname = f"{self.res_dir}/{self.target_class_id}_means.npy"
            print(f"BAGAN: saving multivariate: {cfname}, {mfname}")
            np.save(cfname, self.covariances)
            np.save(mfname, self.means)

    def _get_lst_bck_name(self, element):
        import re
        files = [f for f in os.listdir(self.res_dir) if re.match(rf'bck_c_{self.target_class_id}_{element}', f)]
        if files:
            fname = files[0]
            e_str = os.path.splitext(fname)[0].split("_")[-1]
            epoch = int(e_str)
            return epoch, fname
        return 0, None

    def init_gan(self):
        epoch, generator_fname = self._get_lst_bck_name("generator")
        new_e, discriminator_fname = self._get_lst_bck_name("discriminator")
        if new_e != epoch:
            return 0
        try:
            self.generator.load_weights(os.path.join(self.res_dir, generator_fname))
            self.discriminator.load_weights(os.path.join(self.res_dir, discriminator_fname))
            return epoch
        except:
            return 0

    def backup_point(self, epoch):
        _, old_bck_g = self._get_lst_bck_name("generator")
        _, old_bck_d = self._get_lst_bck_name("discriminator")
        try:
            os.remove(os.path.join(self.res_dir, old_bck_g))
            os.remove(os.path.join(self.res_dir, old_bck_d))
        except:
            pass
        generator_fname = f"{self.res_dir}/bck_c_{self.target_class_id}_generator_e_{epoch}.weights.h5"
        discriminator_fname = f"{self.res_dir}/bck_c_{self.target_class_id}_discriminator_e_{epoch}.weights.h5"
        self.generator.save_weights(generator_fname)
        self.discriminator.save_weights(discriminator_fname)

    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            self.autoenc_epochs = epochs
            self.class_aratio = bg_train.get_class_probability()
            self._set_class_ratios()
            print(f"uratio: {self.class_uratio}, dratio: {self.class_dratio}, gratio: {self.class_gratio}")
            self.init_autoenc(bg_train)
            start_e = self.init_gan()
            print(f"BAGAN gan initialized, start_e: {start_e}")

            crt_c = self.classes[0]
            act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            img_samples = np.array([[
                act_img_samples,
                self.generator.predict(self.reconstructor.predict(act_img_samples, verbose=0), verbose=0),
                self.generate_samples(crt_c, 10, bg_train)
            ]])
            for crt_c in self.classes[1:]:
                act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                new_samples = np.array([[
                    act_img_samples,
                    self.generator.predict(self.reconstructor.predict(act_img_samples, verbose=0), verbose=0),
                    self.generate_samples(crt_c, 10, bg_train)
                ]])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)
            shape = img_samples.shape
            img_samples = img_samples.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))
            save_image_array(img_samples, f'{self.res_dir}/cmp_class_{self.target_class_id}_init.png')

            for e in range(start_e, epochs):
                print(f'GAN train epoch: {e+1}/{epochs}')
                train_disc_loss, train_gen_loss = self._train_one_epoch(bg_train)
                nb_test = bg_test.get_num_samples()
                fake_size = int(np.ceil(nb_test * 1.0 / self.nclasses))
                sampled_labels = self._biased_sample_labels(nb_test, "d")
                latent_gen = self.generate_latent(sampled_labels, bg_test)
                generated_images = self.generator.predict(latent_gen, verbose=0)
                X = np.concatenate((bg_test.dataset_x, generated_images))
                aux_y = np.concatenate((bg_test.dataset_y, np.full(len(sampled_labels), max(self.classes) + 1)), axis=0)
                test_disc_loss = self.discriminator.evaluate(X, aux_y, verbose=0)
                sampled_labels = self._biased_sample_labels(fake_size + nb_test, "g")
                latent_gen = self.generate_latent(sampled_labels, bg_test)
                test_gen_loss = self.combined.evaluate(latent_gen, sampled_labels, verbose=0)
                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                self.test_history['disc_loss'].append(test_disc_loss)
                self.test_history['gen_loss'].append(test_gen_loss)
                print(f"train_disc_loss {train_disc_loss}, train_gen_loss {train_gen_loss}, test_disc_loss {test_disc_loss}, test_gen_loss {test_gen_loss}")

                if e % 10 == 9:
                    img_samples = np.array([self.generate_samples(c, 10, bg_train) for c in self.classes])
                    save_image_array(img_samples, f'{self.res_dir}/plot_class_{self.target_class_id}_epoch_{e}.png')

                if e % 10 == 5:
                    self.backup_point(e)
                    crt_c = self.classes[0]
                    act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    img_samples = np.array([[
                        act_img_samples,
                        self.generator.predict(self.reconstructor.predict(act_img_samples, verbose=0), verbose=0),
                        self.generate_samples(crt_c, 10, bg_train)
                    ]])
                    for crt_c in self.classes[1:]:
                        act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        new_samples = np.array([[
                            act_img_samples,
                            self.generator.predict(self.reconstructor.predict(act_img_samples, verbose=0), verbose=0),
                            self.generate_samples(crt_c, 10, bg_train)
                        ]])
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)
                    shape = img_samples.shape
                    img_samples = img_samples.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))
                    save_image_array(img_samples, f'{self.res_dir}/cmp_class_{self.target_class_id}_epoch_{e}.png')

            self.trained = True

    def generate_samples(self, c, samples, bg=None):
        return self.generate(np.full(samples, c), bg)

    def save_history(self, res_dir, class_id):
        if self.trained:
            filename = f"{res_dir}/class_{class_id}_score.csv"
            generator_fname = f"{res_dir}/class_{class_id}_generator.weights.h5"
            discriminator_fname = f"{res_dir}/class_{class_id}_discriminator.weights.h5"
            reconstructor_fname = f"{res_dir}/class_{class_id}_reconstructor.weights.h5"
            with open(filename, 'w') as csvfile:
                fieldnames = ['train_gen_loss', 'train_disc_loss', 'test_gen_loss', 'test_disc_loss']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for e in range(len(self.train_history['gen_loss'])):
                    row = [
                        self.train_history['gen_loss'][e],
                        self.train_history['disc_loss'][e],
                        self.test_history['gen_loss'][e],
                        self.test_history['disc_loss'][e]
                    ]
                    writer.writerow(dict(zip(fieldnames, row)))
            self.generator.save_weights(generator_fname)
            self.discriminator.save_weights(discriminator_fname)
            self.reconstructor.save_weights(reconstructor_fname)

    def load_models(self, fname_generator, fname_discriminator, fname_reconstructor, bg_train=None):
        self.init_autoenc(bg_train, gen_fname=fname_generator, rec_fname=fname_reconstructor)
        self.discriminator.load_weights(fname_discriminator)