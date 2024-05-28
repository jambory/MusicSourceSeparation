import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers

from parameters import Parameters
    
class Encoder(layers.Layer):
    """ 
    First step of model not counting, preprocessing data. Puts the inputs through a 1D Convolutional layer with N output channels. The encoder also includes a activation
    function of a ReLU layer.

    Args: 
        N: Encoder output size
        win: Kernel size of encoder.

    """
    def __init__(self, param: Parameters):
        super(Encoder, self).__init__(name='Encoder')
        self.U = layers.Conv1D(param.N,param.win,strides=param.overlap,activation="relu",use_bias=False, data_format="channels_last")

    def call(self, x): # (M, T)
        batch_count,sample_count = x.shape
        x = tf.reshape(x, (batch_count, sample_count, 1)) # (M, T, 1)
        return self.U(x) # (M, K, N)
      
class Separator(layers.Layer):
    """ 
    To begin the we take the output of the encoder which we then normalize it and feed it into a 1x1Convolution.
    This layer repeats a Temporal Convolution Block multiple times, wherin, what is refered to as a Conv1D Block in the paper, is repeated
    several times for each instance of a Temporal Convolutional Block. Finally the output of the Temporal blocks is input into a 1x1 Convolution
    with C*N output channels.
    Args:
        N: (int) Encoder output size
        B: (int) Bottleneck Conv1DBlock output size
        R: (int) Amount of repeats of the Temporal Convolution block
        X: (int) Amount of times Conv1DBlock is applied in a Temporal Convolution block
        H: (int) Conv1DBlock input size
        C: (int) Amount of sources being estimated
        P: (int) Size of kernel in depthwise convolutions
    
    """
    def __init__(self, param: Parameters):
        super(Separator, self).__init__(name='Separator')
        self.C = param.C
        self.N = param.N
        self.skip=param.skip
        if param.casual:
            self.normalization = cLN(param.N)
        else:
            self.normalization = gLN(param.N) 
        self.bottle_neck = layers.Conv1D(param.B,1)
        
        self.temporal_conv = [TemporalConv(param.X, param.H, param.B, param.P, param.casual, skip=param.skip) for r in range(param.R)]
        self.skip_add = layers.Add()
        self.prelu = layers.PReLU()
        self.m_i = layers.Conv1D(param.C*param.N, 1)

    def call(self, w): # (M, K, N)
        M, K, _ = w.shape
        normalized_w = self.normalization(w) # (M, K, N)
        output = self.bottle_neck(normalized_w) # (M, K, B)
        if self.skip:
            skip_conn = tf.zeros(output.shape)
        for block in self.temporal_conv:
            if self.skip:
                output, skips = block(output)
                skip_conn = self.skip_add([skip_conn, skips])
            else:
                output = block(output)
        # (M, K, B)
        if self.skip:
            output = skip_conn

        output = self.prelu(output)
        estimated_masks = self.m_i(output) # (M, K, C*N)
        estimated_masks = activations.sigmoid(tf.reshape(estimated_masks, (M, self.C, K, self.N))) # (M, C, K, N)
        return estimated_masks
    
class Decoder(layers.Layer):
    """ 
    This is layer is final part of the model. Taking the features obtained, from the Conv1D blocks masks are finally combined into with the
    original mix into the estimated sources.

    Args:
        L : (int) length of individual chunks of audio.
    """
    def __init__(self, param: Parameters):
        super(Decoder, self).__init__(name='Decoder')
        self.C = param.C
        self.L = param.L
        self.N = param.N
        self.win = param.win
        self.naplab_impl = param.naplab_impl
        self.overlap = param.overlap

        if not self.naplab_impl:
            self.transpose_conv = layers.Conv1DTranspose(self.L,self.win,strides=param.overlap,use_bias=False)
        else:
            self.transpose_conv = layers.Conv1DTranspose(1,self.win,strides=param.overlap,use_bias=False)

    def call(self, w, m_i): # (M, K, N) (M, C, K, N)
        M,K,_, = w.shape

        w = tf.reshape(tf.repeat(w, 4, axis=0),m_i.shape) # (M, K, N) -> (M, C, K, N)
        output = w * m_i # (M, C, K, N) * (M, C, K, N) 
        output = self.transpose_conv(tf.reshape(output, (M*self.C, K, self.N))) # (M*C, T, 1) or (M*C, T, L)Okay so the official naplab implementation turns the encoded sources directly back into its wavform essentially 
                                                                                # but the kaituoxo one seems to turn it into a 16 channel output then subsetting to only take the first chunk to deal with overlap
        if not self.naplab_impl:
            output = output[:,:,:self.overlap]
        output = tf.reshape(output, (M,self.C,-1)) # (M, C, T)
        return output   

class Conv1DBlock(layers.Layer):
    def __init__(self, H, B, P, dilation, casual, skip = True):
        super(Conv1DBlock, self).__init__(name='Conv1DBlock')
        self.skip = skip
        self.input_channels = layers.Conv1D(H,1)
        self.prelu = layers.PReLU()
        if casual:
            self.norm = cLN(H)
        else:
            self.norm = gLN(H)
        
        self.depthwise = DepthwiseConv(H, B, P, dilation, casual, skip=self.skip)
        self.res_add = layers.Add()

    def call(self, input):
        input_channels = self.input_channels(input)
        input_channels = self.prelu(input_channels)
        input_channels = self.norm(input_channels)
        if self.skip:
            res, skip = self.depthwise(input_channels)
            res = self.res_add([res, input])
            return res, skip

        res = self.depthwise(input_channels, skip=self.skip)
        res = self.res_add([res, input])
        return res

class DepthwiseConv(layers.Layer):
    def __init__(self, H, B, P, dilation, casual, skip=True):
        super(DepthwiseConv, self).__init__(name='DepthwiseConv')
        self.skip = skip
        if casual:
            padding_type = "causal"
            self.norm = cLN(H)
        else:
            padding_type = "same"
            self.norm = gLN(H)

        self.conv1d = layers.Conv1D(H, P, dilation_rate=dilation, padding=padding_type, groups=H)
        self.prelu = layers.PReLU()
        
        self.res_out = layers.Dense(B)
        if self.skip:
            self.skip_out = layers.Dense(B)
    
    def call(self, input):
        output = self.conv1d(input) 
        output = self.prelu(output)
        output = self.norm(output)
        res = self.res_out(input)
        if self.skip:
            skip_out = self.skip_out(output)
            return res, skip_out
        
        return res

class TemporalConv(layers.Layer):
    def __init__(self, X, H, B, P, casual, skip=True):
        super(TemporalConv,self).__init__(name='TemporalConv')
        self.skip=skip
        self.blocks = []
        for i in range(X):
            dilation = 2 ** i
            self.blocks += [Conv1DBlock(H, B, P, dilation, casual, skip=self.skip)]
        self.res_add = layers.Add()
        self.skip_add = layers.Add()
    
    def call(self, block_input):
        if self.skip:
            skip_conn = tf.zeros(block_input.shape)
        for block in self.blocks:    
                if self.skip:
                    res, skip = block(block_input)
                    block_input = self.res_add([block_input, res])
                    skip_conn = self.skip_add([skip_conn,skip])
                else:
                    res = block(block_input)
                    block_input = self.res_add([block_input, res])
        if self.skip:
            return block_input, skip_conn
        return block_input
        

class cLN(layers.Layer):
    def __init__(self, H, EPS = 1e-8):
        super(cLN, self).__init__(name="cLN")
        self.EPS = EPS
        shape = (1,1,H)
        self.gamma = tf.Variable(tf.ones(shape),trainable=True,shape=shape,name="Casual Gamma")
        self.beta = tf.Variable(tf.zeros(shape),trainable=True,shape=shape,name="Casual Beta")

    def call(self, input):
        # (M, K, H)
        E_f = tf.math.reduce_mean(input, axis=2, keepdims=True) # (M, K, 1)
        var = tf.math.reduce_variance(input, axis=2, keepdims=True)# (M, K, 1)

        normalized = self.gamma * (input - E_f) / tf.math.sqrt(var + self.EPS) + self.beta
        # (M, K, H)
        return normalized
    
class gLN(layers.Layer):
    def __init__(self, H, EPS = 1e-8):
        super(gLN, self).__init__(name="gLN")
        self.EPS = EPS
        shape = (1,1,H)
        self.gamma = tf.Variable(tf.ones(shape),trainable=True,shape=shape,name="Global Gamma")
        self.beta = tf.Variable(tf.zeros(shape),trainable=True,shape=shape,name="Global Beta")

    def call(self, input):
        # (M, K, H)
        E_f = tf.math.reduce_mean(input, axis=(1,2), keepdims=True) # (M, 1, 1)
        var = tf.math.reduce_variance(input, axis=(1,2), keepdims=True)# (M, 1, 1)

        normalized = self.gamma * (input - E_f) / tf.math.sqrt(var + self.EPS) + self.beta
        # (M, K, H)
        return normalized
