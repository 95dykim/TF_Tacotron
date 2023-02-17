import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

os.environ["TFDS_DATA_DIR"] = "./cache/"

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd

import shutil
import librosa
import pickle
import glob

CHAR_SEQ_LEN = 20

ENCODER_CBHG_K = 16
ENCODER_CBHG_CHANNEL = 128

def CBHG(x_o, channels, K, name=""):
    # Conv 1D Bank + stacking
    list_x_k = []
    for idx_K in range(K):
        x_k = tf.keras.layers.Conv1D(128, idx_K+1, padding="same", use_bias=False, name=name+"_ConvBank_{}_conv".format(idx_K+1))(x_o)
        x_k = tf.keras.layers.BatchNormalization(name=name+"_ConvBank_{}_bn".format(idx_K+1))(x_k)
        x_k = tf.keras.layers.Activation("relu", name=name+"_ConvBank_{}_relu".format(idx_K+1))(x_k)
        list_x_k.append(x_k)
    x = tf.keras.layers.Concatenate(axis=-1, name=name+"_stack")(list_x_k)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same', name=name+"_pool")(x)
    
    # Conv1D projections
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False, name=name+"_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=name+"_relu1")(x)
    
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False,  name=name+"_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_bn2")(x)
    
    # Residual connection
    x = tf.keras.layers.Add(name=name+"_residual")([x, x_o])
    
    # Highway layers
    # https://paperswithcode.com/method/highway-layer
    # 왜 TACTRON 페이퍼는 레이어 4개? 하이웨이 레이어 2번 거치나? -> Highway Layers니까 맞는듯
    for idx in range(2):
        h = tf.keras.layers.Dense(128, name=name+"_highway_h{}_dense".format(idx+1))(x)
        h = tf.keras.layers.Activation("relu", name=name+"_highway_h{}_relu".format(idx+1))(h)

        t = tf.keras.layers.Dense(128, name=name+"_highway_t{}_dense".format(idx+1))(x)
        t = tf.keras.layers.Activation("sigmoid", name=name+"_highway_t{}_sigmoid".format(idx+1))(t)

        x = t*h + (1-t)*x
    
    x = tf.keras.layers.Bidirectional( tf.keras.layers.GRU(128), name=name+"_BiGRU" )(x)
    
    return x
    
#Model Input
inputs = tf.keras.layers.Input(shape=(CHAR_SEQ_LEN,), name="inputs" )

#Embedding
x = tf.keras.layers.Embedding(CHAR_SEQ_LEN, 256, name="CharEmbedding")(inputs)

#Encoder pre-net
x = tf.keras.layers.Dense(256, name="Encoder_prenet1_dense")(x)
x = tf.keras.layers.Activation("relu", name="Encoder_prenet1_relu")(x)
x = tf.keras.layers.Dropout(0.5, name="Encoder_prenet1_dropout")(x)
x = tf.keras.layers.Dense(128, name="Encoder_prenet2_dense")(x)
x = tf.keras.layers.Activation("relu", name="Encoder_prenet2_relu")(x)
x = tf.keras.layers.Dropout(0.5, name="Encoder_prenet2_dropout")(x)

#Encoder CBHG
x = CBHG( x, ENCODER_CBHG_CHANNEL, ENCODER_CBHG_K, "Encoder_CBHG" )


################################################################
# DECODER

#Decoder pre-net

# pre-net should be made into a sequential model
x = tf.keras.layers.Dense(256, name="Decoder_prenet1_dense")
x = tf.keras.layers.Activation("relu", name="Decoder_prenet1_relu")
x = tf.keras.layers.Dropout(0.5, name="Decoder_prenet1_dropout")
x = tf.keras.layers.Dense(128, name="Decoder_prenet2_dense")
x = tf.keras.layers.Activation("relu", name="Decoder_prenet2_relu")
x = tf.keras.layers.Dropout(0.5, name="Decoder_prenet2_dropout")

class RNNWrapper_Residual(tf.keras.layers.AbstractRNNCell):
    def __init__(self, cell, **kwargs):
        super(RNNWrapper_Residual, self).__init__(**kwargs)
        self.cell_wrapped = cell

    @property
    def state_size(self):
        return self.cell_wrapped.units

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        cell_out, cell_states = self.cell_wrapped(inputs, states)

        return cell_out + cell_states[0], cell_states

################################################################

#Vinyals et al. (2015), a content-based tanh attention decoder
#https://arxiv.org/pdf/1412.7449.pdf

#참고
#https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/seq2seq/attention_wrapper.py#L1565-L2063
#-> 구혀낭 
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/AbstractRNNCell

# 만들어야하는거 - Residual RNN, Attention RNN wrapper

outputs = x
