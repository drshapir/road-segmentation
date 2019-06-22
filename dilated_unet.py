#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro
"""
from utils import bce_dice_loss, dice_coef
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop


class DilatedUNet(object):
    def __init__(self, n_blocks=3, filters=32, input_shape=(512, 512, 3),
                 lr=1e-4, loss=bce_dice_loss, optimizer=RMSprop):
        self.n_blocks = n_blocks
        self.filters = filters
        self.input_shape = input_shape
        self.lr = lr
        self.loss = loss
        self.optimizer = optimizer
        self.encoders = []

    def _encoder(self, input_layer, filters):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        self.encoders.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x

    def _decoder(self, input_layer, skip, filters):
        x = UpSampling2D(size=(2, 2))(input_layer)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = concatenate([skip, x])
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    def bottleneck(self, x, filters=512, depth=6):
        dilations = []
        for i in range(depth):
            x = Conv2D(filters, (3, 3), activation='relu', padding='same',
                       dilation_rate=2**i)(x)
            dilations.append(x)
        return add(dilations)

    def compile_model(self):
        inputs = Input(self.input_shape)
        x = inputs
        for i in range(self.n_blocks):
            x = self._encoder(x, self.filters * 2**i)
        x = self.bottleneck(x, self.filters * 2**self.n_blocks)
        for i in reversed(range(self.n_blocks)):
            x = self._decoder(x, self.encoders[i], self.filters * 2**i)
        output = Conv2D(1, 1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=self.optimizer(self.lr), loss=self.loss,
                      metrics=[dice_coef])
        return model
