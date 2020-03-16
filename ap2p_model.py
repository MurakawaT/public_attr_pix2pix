import numpy as np
import scipy
import tensorflow
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras_vggface.vggface import VGGFace
import datetime
import os
from PIL import Image

from tensorflow.keras.optimizers import Adam


class Ap2p():
    """
    Build Attr_Pix2pix class
    """
    def __init__(self):

        # Input shape
        self.input_img_rows = 128
        self.input_img_cols = 128
        self.channels = 3
        self.gene_attr_shape = (1, 1, 4)
        self.discri_attr_shape = (16, 16, 4)
        self.input_img_shape = (self.input_img_rows, self.input_img_cols, self.channels)
        self.input_img_low_shape = (self.input_img_rows//8, self.input_img_cols//8, self.channels)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='mse',
            optimizer=optimizer
        )

        # Build the generator
        self.generator = self.build_generator()

        # Input
        img_low_source =  Input(shape=self.input_img_low_shape)
        gene_attr = Input(shape = self.gene_attr_shape)
        discri_attr = Input(shape = self.discri_attr_shape)

        # convert source image into fake image
        img_fake = self.generator([img_low_source, gene_attr])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of (fake)
        valid = self.discriminator([img_fake, discri_attr])

        # face_netで顔画像をベクトルに
        self._vggface = VGGFace(include_top=False, input_shape=self.input_img_shape)
        self._vggface.trainable = False
        face_vec_fake = self._vggface(img_fake)

        self.combined = Model(inputs=[img_low_source, gene_attr, discri_attr], outputs=[valid, face_vec_fake, img_fake])
        self.combined.compile(loss=['mae', 'mae','mae'],
                              loss_weights=[0.95, 0.95, 1],
                              optimizer=optimizer)

    def build_generator(self):
        """
        U-Net Generator
        generator: input image(16*16) -> output image(128*128)
        """

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.input_img_low_shape) # (n, 16, 16, 3)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False) #(n, 8, 8, 32)
        d2 = conv2d(d1, self.gf*2) #(n, 4, 4, 64)
        d3 = conv2d(d2, self.gf*4) #(n, 2, 2, 128)
        d4 = conv2d(d3, self.gf*8) #(n, 1, 1, 256)

        # attr input ここで属性情報を挿入
        input_attr = Input(shape = self.gene_attr_shape)
        d4_attr = Concatenate()([d4, input_attr]) #(n, 1, 1, 260)

        # Upsampling
        u1 = deconv2d(d4_attr, d3, self.gf*8) #(n, 2, 2, 256)
        u2 = deconv2d(u1, d2, self.gf*8)#(n, 4, 4, 256)
        u3 = deconv2d(u2, d1, self.gf*8)#(n, 8, 8, 256)
        u4 = deconv2d(u3, d0, self.gf*8)#(n, 16, 16, 256)

        u5 = UpSampling2D(size=2)(u4)#(n, 32, 32, 256)
        u5_1 = Conv2D(self.gf*4, kernel_size=3, strides=1, padding='same')(u5)
        u5_2 = LeakyReLU(alpha=0.2)(u5_1)
        #(n, 32, 32, 128)

        u6 = UpSampling2D(size=2)(u5_2)#(n, 64, 64, 128)
        u6_1 = Conv2D(self.gf*2, kernel_size=3, strides=1, padding='same')(u6)
        u6_2 = LeakyReLU(alpha=0.2)(u6_1)
        #(n, 64, 64, 64)

        u7 = UpSampling2D(size=2)(u6_2)#(n, 64, 64, 128)
        u7_1 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(u7)
        u7_2 = LeakyReLU(alpha=0.2)(u7_1)
        #(n, 128, 128, 32)

        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7_2)

        return Model([d0, input_attr], [output_img])

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_source = Input(shape=self.input_img_shape) #(128, 128, 3)

        d1 = d_layer(img_source, self.df, bn=False) #(64, 64, 32)
        d2 = d_layer(d1, self.df*2) #(32, 32, 64)
        d3 = d_layer(d2, self.df*4) #(16, 16, 128)

        # attr input　ここに属性情報を挿入
        input_attr = Input(shape = self.discri_attr_shape)
        d3_attr = Concatenate()([d3, input_attr])

        d4 = d_layer(d3_attr, self.df*8) #(None, 8, 8, 256)
        d5 = d_layer(d4, self.df*8) #(None, 4, 4, 256)
        d6 = d_layer(d5, self.df*8) #(None, 2, 2, 256)
        d7 = Flatten()(d6) #(None, 2*2*512)
        d8 = Dense(1024)(d7)
        validity = Dense(1, activation='tanh')(d8)

        return Model([img_source, input_attr], validity)

    def train(self, get_imgs_source, attr_flag_valid):
        batch_size = len(get_imgs_source)

        #画像をnmupy配列に (-1,1の範囲)
        imgs_source = np.empty((0, 128, 128, 3))
        imgs_low_source =  np.empty((0, 16, 16, 3))
        for i in range(0,batch_size):
            #high　高解像度画像
            img_source = (np.asarray(get_imgs_source[i]) - 127.5) / 127.5
            img_source = img_source.reshape(-1, 128, 128, 3)
            imgs_source = np.concatenate([imgs_source, img_source], axis = 0)
            #low　低解像度画像
            img_low_source = get_imgs_source[i].resize((16,16),Image.BICUBIC)
            img_low_source = (np.asarray(img_low_source) - 127.5) / 127.5
            img_low_source = img_low_source.reshape(-1, 16, 16, 3)
            imgs_low_source = np.concatenate([imgs_low_source, img_low_source], axis = 0)

        #attr_valid　正しい属性情報を変形
        vec_valid = np.zeros((batch_size, 1)) + attr_flag_valid.reshape((batch_size, 1))
        attr_gene_valid = np.zeros((batch_size,)+ self.gene_attr_shape) + attr_flag_valid.reshape((batch_size,1,1,1))
        attr_discri_valid = np.zeros((batch_size,)+ self.discri_attr_shape) + attr_flag_valid.reshape((batch_size,1,1,1))

        #　偽の属性情報を生成
        attr_flag_fake = 1 - attr_flag_valid
        #　偽の属性情報を変形
        vec_fake = np.zeros((batch_size, 1)) + attr_flag_fake.reshape((batch_size, 1))
        attr_discri_fake = np.zeros((batch_size,)+ self.discri_attr_shape) + attr_flag_fake.reshape((batch_size,1,1,1))

        # generatorでfake画像を作る
        imgs_fake = self.generator.predict([imgs_low_source,attr_gene_valid])

        # train　discriminator 
        d_loss_real = self.discriminator.train_on_batch([imgs_source, attr_discri_valid], vec_valid) #(h,a)
        d_loss_fake_1 = self.discriminator.train_on_batch([imgs_source, attr_discri_fake], vec_fake) #(h,~a)
        d_loss_fake_2 = self.discriminator.train_on_batch([imgs_fake, attr_discri_valid], vec_fake) #(~h,a)

        # 真の顔ベクトル生成
        face_vec_valid = self._vggface.predict(imgs_source)

        # Train generator
        g_loss = self.combined.train_on_batch(
            [imgs_low_source, attr_gene_valid, attr_discri_valid], [vec_valid, face_vec_valid, imgs_source]
            )# (~h,a)
        
        losses = (d_loss_real, d_loss_fake_1, d_loss_fake_2, g_loss[0])

        return losses
    

    def pred(self, get_imgs_source, strength):
        batch_size = len(get_imgs_source)

        #画像をnmupy配列に (-1,1の範囲)
        imgs_source = np.empty((0, 128, 128, 3))
        imgs_low_source =  np.empty((0, 16, 16, 3))
        for i in range(0,batch_size):
            #high
            img_source = (np.asarray(get_imgs_source[i]) - 127.5) / 127.5
            img_source = img_source.reshape(-1, 128, 128, 3)
            imgs_source = np.concatenate([imgs_source, img_source], axis = 0)
            #low
            img_low_source = get_imgs_source[i].resize((16,16),Image.BICUBIC)
            img_low_source = (np.asarray(img_low_source) - 127.5) / 127.5
            img_low_source = img_low_source.reshape(-1, 16, 16, 3)
            imgs_low_source = np.concatenate([imgs_low_source, img_low_source], axis = 0)
        
        #attr_valid　属性情報を変形
        attr_flag_valid = np.ones(imgs_source.shape[0])
        attr_gene_valid = np.zeros((batch_size,)+ self.gene_attr_shape) + attr_flag_valid.reshape((batch_size,1,1,1))
        #属性の強度を上げる
        attr_gene_valid *= strength

        #　もう一つの属性情報を生成
        attr_flag_fake = 1 - attr_flag_valid
        #　変形
        attr_gene_fake = np.zeros((batch_size,)+ self.gene_attr_shape) + attr_flag_fake.reshape((batch_size,1,1,1))

        #　それぞれの属性で超解像処理
        imgs_fake_attr_valid = self.generator.predict([imgs_low_source,attr_gene_valid])
        imgs_fake_attr_fake = self.generator.predict([imgs_low_source,attr_gene_fake])

        return (imgs_fake_attr_valid, imgs_fake_attr_fake)


    def save_weights(self, file_path, overwrite=True):
        self.combined.save_weights(file_path + "_combine.h5", overwrite)
        self.generator.save_weights(file_path + "_generator.h5", overwrite)
        self.discriminator.save_weights(file_path + "_discriminator.h5", overwrite)
    
    def load_weights(self, file_path, by_name=False):
        self.combined.load_weights(file_path + "_combine.h5")
        self.generator.load_weights(file_path + "_generator.h5")
        self.discriminator.load_weights(file_path + "_discriminator.h5")
