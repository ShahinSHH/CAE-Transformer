# -*- coding: utf-8 -*-
"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model

def build_encoder(input_shape = (512,512,1), conv_dims = [32,64,128], dense_dims = [1024], activation = 'relu', show_summary = True):  
    
    encoder_inputs = Input(shape=input_shape)
    
    x = Conv2D(conv_dims[0], 3, activation=activation, strides=1, padding="same", trainable = True)(encoder_inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    if len(conv_dims)>1:
        for  dim in conv_dims[1::]:
            x = Conv2D(dim, 3, activation=activation, strides=1, padding="same", trainable = True)(x)
            x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)
    x = Dense(dense_dims[0], activation=activation, trainable = True)(x)
    if len(dense_dims) >1:
        for dens in dense_dims[1::]:
            x = Dense(dens, activation=activation, trainable = True)(x)
    
    encoder = Model(inputs = encoder_inputs, outputs = x, name="encoder")
    if show_summary:
        encoder.summary()
    return encoder


def build_encoder_with_loc(input_shape = (512,512,1), conv_dims = [32,64,128], dense_dims = [1024], activation = 'relu', show_summary = True):  
    
    encoder_inputs = Input(shape=input_shape)
    
    x = Conv2D(conv_dims[0], 3, activation=activation, strides=1, padding="same", trainable = True)(encoder_inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    if len(conv_dims)>1:
        for  dim in conv_dims[1::]:
            x = Conv2D(dim, 3, activation=activation, strides=1, padding="same", trainable = True)(x)
            x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)
    x = Dense(dense_dims[0], activation=activation, trainable = True)(x)
    if len(dense_dims) >1:
        for dens in dense_dims[1::]:
            x = Dense(dens, activation=activation, trainable = True)(x)
    
    encoder = Model(inputs = encoder_inputs, outputs = x, name="encoder")
    if show_summary:
        encoder.summary()
    return encoder

def build_decoder(output_size = 512, deconv_dims = [128,64,32], dense_dims = [1024], activation = 'relu', show_summary = True):
    latent_inputs = Input(shape=(dense_dims[0]))
    num_deconv = len(deconv_dims)
    dense_neurons = int(output_size/(2**num_deconv)) # it is a quotient of the output size, so that the image will be reconstructed to the right size correctly
    x = Dense(dense_neurons * dense_neurons * dense_dims[0], activation=activation, trainable = True)(latent_inputs)
    
    x = Reshape((dense_neurons, dense_neurons, dense_dims[0]))(x)
    
    x = Conv2DTranspose(deconv_dims[0], 3, activation=activation, strides=2, padding="same", trainable = True)(x)
    if num_deconv >1:
            for dim in deconv_dims[1::]:
                x = Conv2DTranspose(dim, 3, activation=activation, strides=2, padding="same", trainable = True)(x)
    
    decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same", trainable = True)(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    if show_summary:
        decoder.summary()
    return decoder


class CAE(Model):
    def __init__(self,
                 encoder,
                 decoder,
                 model_input_shape = (512,512,1),
                 **kwargs):
        super(CAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.model_input_shape = model_input_shape

    def call(self, inputs):
        
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs

    def model(self):
        
        x = Input(shape = self.model_input_shape)
        return Model(inputs=[x], outputs=self.call(x))   

