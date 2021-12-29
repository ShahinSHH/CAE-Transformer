# -*- coding: utf-8 -*-
"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
Implementation of the transformer is adopted from the implementation of ViT by Khalid Salama,
available at: https://keras.io/examples/vision/image_classification_with_vision_transformer/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, layers
from tensorflow.keras.backend import *
K.set_image_data_format('channels_last')

       
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_patches':  self.num_patches,
        'projection' : self.projection,                  
        })
        return config

class PositionEncoder(Layer): # only position encoder without patch encoder
    def __init__(self, num_patches, projection_dim):
        super(PositionEncoder, self).__init__()
        self.num_patches = num_patches
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = x + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_patches':  self.num_patches,              
        })
        return config

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches  
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'patch_size':  self.patch_size,                 
        })
        return config
    
    
class Transformer(Model):
    def __init__(self,
             feature_size = 256,
             num_slices = 25,
             projection_dim = 256,
             num_heads = 5,
             transformer_layers = 3,
             mlp_head_units = [32],
             num_classes = 2,
             trainable = True,
             dense_activation = 'relu',
             transformer_dropout = 0,
             mlp_dropout = 0,
             fc_dropout = 0,
             key_dim = 128,
             noise = 0,
             attention_axis = None,
             **kwargs):
          # call the parent constructor
          super(Transformer, self).__init__(**kwargs)
          
          # initializing parameters
          self.feature_size = feature_size
          self.num_slices = num_slices
          self.projection_dim = projection_dim
          self.num_heads = num_heads
          self.transformer_layers = transformer_layers
          self.mlp_head_units = mlp_head_units
          self.num_classes = num_classes
          self.trainable = trainable
          self.transformer_dropout = transformer_dropout
          self.mlp_dropout = mlp_dropout
          self.fc_dropout = fc_dropout
          self.dense_activation = dense_activation
          self.noise = noise 
          self.transformer_units = [self.projection_dim , self.projection_dim] # different from the ViT
          self.model_input_shape = (self.num_slices, self.feature_size)
          self.key_dim = key_dim # key and query dimensions are provided by the input
          self.attention_axis = attention_axis
    
    def mlp(self, x, hidden_units):
        for units in hidden_units:
            x = Dense(units, activation=self.dense_activation)(x)
            x = Dropout(self.mlp_dropout)(x)
        return x
    
    def Transformer_Classifier(self, x):
        # inputs = Input(shape=input_shape)
        # Augment data.
        # augmented = data_augmentation(inputs)
        # Create patches.
        # Create multiple layers of the Transformer block.
        for n_layer in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = LayerNormalization(epsilon=1e-12)(x)
            # Adding Gaussian noise (optional)
            # Create a multi-head attention layer.
            if n_layer == -1: # considering an exception for the first layer to adjust the feature size for the next layers
            # to put it in function, change -1 to 0, otherwise it is always FALSE
                attention_output = MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim = self.key_dim,
                    value_dim = self.feature_size,
                    dropout=self.transformer_dropout,
                    attention_axes = self.attention_axis
                )(x1, x1)
                
            else:
                 attention_output = MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.key_dim,
                    value_dim = self.projection_dim,
                    dropout=self.transformer_dropout,
                    attention_axes = self.attention_axis
                )(x1, x1)   
            # Skip connection 1.
            x2 = Add()([attention_output, x])
            # Layer normalization 2.
            x3 = LayerNormalization(epsilon=1e-12)(x2)
            # MLP.
            if n_layer == -1: # if the first layer should be adjusted, we can't have a residual connection
                x = self.mlp(x3, hidden_units=self.transformer_units)
            else: 
                # Skip connection 2.
                x3 = self.mlp(x3, hidden_units=self.transformer_units)
                x = Add()([x3, x2])
                # x = GaussianNoise(self.noise)(x)

        # Create a [batch_size, projection_dim] tensor.
        representation = LayerNormalization(epsilon=1e-12)(x)
        """
        There are multiple options to aggregate the generated feature by the transformer.
        In the CAE-Transformer, Global Max Pooling provides the best results.
        You can change it to Global Average Pooling or Flatten by uncommenting one of the following lines.
        """
        # representation = Flatten()(representation) # Flatten the features across the time instances
        # representation = GlobalAveragePooling1D()(representation) # Average the features across the instances
        representation = GlobalMaxPooling1D()(representation) # Take the maximum of the features across the instances
        
        representation = Dropout(self.fc_dropout)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units)
        # Classify outputs.
        logits = Dense(self.num_classes, activation='softmax')(features)
        return logits
    
    def call(self, inputs):
            x = PositionEncoder(num_patches = self.num_slices, projection_dim = self.projection_dim)(inputs) # using positioal encoding
            x = self.Transformer_Classifier(x)
            return x
    
    def model(self):
            x = Input(shape = self.model_input_shape)
            return Model(inputs=[x], outputs=self.call(x)) 
           
        
class Transformer_Radiomics(Model):
    def __init__(self,
             feature_size = 256,
             num_slices = 25,
             projection_dim = 256,
             num_heads = 5,
             transformer_layers = 3,
             num_radiomics_features = 47, # might be changed during the class instantiation for different versions of the Radiomics data
             mlp_head_units = [32],
             num_classes = 2,
             trainable = True,
             dense_activation = 'relu',
             transformer_dropout = 0,
             mlp_dropout = 0,
             fc_dropout = 0,
             key_dim = 128,
             noise = 0,
             attention_axis = None,
             **kwargs):
          # call the parent constructor
          super(Transformer_Radiomics, self).__init__(**kwargs)
          # initializing parameters
          
          self.feature_size = feature_size
          self.num_slices = num_slices
          self.projection_dim = projection_dim
          self.num_heads = num_heads
          self.transformer_layers = transformer_layers
          self.mlp_head_units = mlp_head_units
          self.num_classes = num_classes
          self.trainable = trainable
          self.transformer_dropout = transformer_dropout
          self.mlp_dropout = mlp_dropout
          self.fc_dropout = fc_dropout
          self.dense_activation = dense_activation
          self.noise = noise
          self.num_radiomics_features = num_radiomics_features  
          self.transformer_units = [self.projection_dim , self.projection_dim] # different from the ViT
          self.model_input_shape = (self.num_slices, self.feature_size)
          self.key_dim = key_dim # key and query dimensions are provided by the input
          self.attention_axis = attention_axis
    def mlp(self, x, hidden_units):
        for units in hidden_units:
            x = Dense(units, activation=self.dense_activation)(x)
            x = Dropout(self.mlp_dropout)(x)
        return x
    
    def Transformer_Classifier(self, x):
        # Create multiple layers of the Transformer block.
        for n_layer in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = LayerNormalization(epsilon=1e-12)(x)
            # Create a multi-head attention layer.
            if n_layer == -1: # considering an exception for the first layer to adjust the feature size for the next layers
            # to put it in function, change -1 to 0, otherwise it is always FALSE
                attention_output = MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim = self.key_dim,
                    value_dim = self.feature_size,
                    dropout=self.transformer_dropout,
                    attention_axes = self.attention_axis
                )(x1, x1)
                
            else:
                 attention_output = MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.key_dim,
                    value_dim = self.projection_dim,
                    dropout=self.transformer_dropout,
                    attention_axes = self.attention_axis
                )(x1, x1)  
                 
            # Skip connection 1.
            x2 = Add()([attention_output, x])
            # Layer normalization 2.
            x3 = LayerNormalization(epsilon=1e-12)(x2)
            # MLP.
            
            if n_layer == -1: # if the first layer should be adjusted, we can't have a residual connection
                # x = GaussianNoise(self.noise)(x)    
                x = self.mlp(x3, hidden_units=self.transformer_units)
            else: 
                # Skip connection 2.
                # x3 = GaussianNoise(self.noise)(x3)
                x3 = self.mlp(x3, hidden_units=self.transformer_units)
                x = Add()([x3, x2])
                
        # Create a [batch_size, projection_dim] tensor.
        """
        There are multiple options to aggregate the generated feature by the transformer.
        In the CAE-Transformer, Global Max Pooling provides the best results.
        You can change it to Global Average Pooling or Flatten by uncommenting one of the following lines.
        """
        representation = LayerNormalization(epsilon=1e-12)(x)
        # representation = Flatten()(representation) # Flatten the features across the time instances
        # representation = GlobalAveragePooling1D()(representation) # Average the features across the instances
        representation = GlobalMaxPooling1D()(representation) # Take the maximum of the features across the instances
        # representation = Dropout(self.fc_dropout)(representation)
        return representation
    
    def call(self, inputs):
           
            x = PositionEncoder(num_patches = self.num_slices, projection_dim = self.projection_dim)(inputs[0]) # using positioal encoding
            x = self.Transformer_Classifier(x)

            features = concatenate([x, inputs[1]], axis=-1)
            features = Dropout(self.fc_dropout)(features)
            # Add MLP.
            features = self.mlp(features, hidden_units=self.mlp_head_units)            
            # Classify outputs.
            outputs = Dense(self.num_classes, activation='softmax')(features)

            return outputs
    
    def model(self):
            x1 = Input(shape = self.model_input_shape) # input 1
            x2 = Input(shape = self.num_radiomics_features) # input 2
            return Model(inputs=[x1, x2], outputs=self.call([x1,x2])) 