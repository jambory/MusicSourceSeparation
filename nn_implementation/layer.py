import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers

class Encoder(layers.Layer):
    """ 
    First step of model not counting, preprocessing data. Puts the inputs through a 1x1 Convolutional layer with N output channels.
    A 1x1Convolutional layer can be interpreted and implemented as a Dense layer, but I chose not to implement it this way for clarity.
    The layer also has a ReLU activation function attached to it.

    Args: 
        N: Encoder output size

    """
    def __init__(self, N):
        super(Encoder, self).__init__(name='Encoder')
        self.U = layers.Conv1D(N,1)

    def call(self, x): # (M, K, L)
        return self.U(x) # (M, K, N)

        
class Separator(layers.Layer):
    """ 
    To begin the we take the output of the encoder which we then normalize it and feed it into a 1x1Convolution.
    This layer repeats a Temporal Convolution Block multiple times, wherin, what is refered to as a Conv1D Block in the paper, is repeated
    several times for each instance of a Temporal Convolutional Block. Finally the output of the Temporal blocks is input into a 1x1 Convolution
    with C*N output channels.
    Args:
        N: (int) Encoder output size
        B: (int) Conv1DBlock output size
        R: (int) Amount of repeats of the Temporal Convolution block
        X: (int) Amount of times Conv1DBlock is applied in a Temporal Convolution block
        H: (int) Conv1DBlock input size
        C: (int) Amount of sources being estimated
    
    """
    def __init__(self, N, B, R, X, H, C, P, casual, skip=True):
        super(Separator, self).__init__(name='Separator')
        self.C = C
        self.N = N
        self.skip=skip
        if casual:
            self.normalization = cLN(N)
        else:
            self.normalization = gLN(N) 
        self.bottle_neck = layers.Conv1D(B,1)
        
        self.temporal_conv = [TemporalConv(X, H, B, P, casual, skip=self.skip) for r in range(R)]
        self.skip_conn = layers.Add()
        self.prelu = layers.PReLU()
        self.m_i = layers.Conv1D(C*N, 1)

    def call(self, w): # (M, K, N)
        normalized_w = self.normalization(w) # (M, K, N)
        output = self.bottle_neck(normalized_w) # (M, K, B)
        skip_list = []
        for i,block in enumerate(self.temporal_conv):
            if self.skip:
                output, skips = block(output)
                skip_list.append(skips)
            else:
                output = block(output)
        # (M, K, B)
        if self.skip:
            output = self.skip_conn(skip_list)

        output = self.prelu(output)
        estimated_masks = self.m_i(output) # (M, K, C*N)
        M, K, _ = estimated_masks.shape
        estimated_masks = activations.sigmoid(tf.reshape(estimated_masks, (M, self.C, K, self.N)))
        return estimated_masks
    
class Decoder(layers.Layer):
    """ 
    This is layer is final part of the model. Taking the features obtained, from the Conv1D blocks masks are finally combined into with the
    original mix into the estimated sources.

    Args:
        L : (int) length of individual chunks of audio.
    """
    def __init__(self, L):
        super(Decoder, self).__init__(name='Decoder')
        self.L = L
        self.transpose_conv = layers.Conv1DTranspose(L,1)

    def call(self, m_i, w):
        M, C, K, _ = m_i.shape
        est_sources =[]
        for i in range(C):
            source_mask = m_i[:,i,:,:]
            masked_source = w * source_mask
            est_src = self.transpose_conv(masked_source)
            est_sources.append(tf.reshape(est_src, (M, 1, K, self.L)))

        decoded_outputs = tf.concat(est_sources, axis=1)
        return decoded_outputs      
    
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

    def call(self, input):
        input_channels = self.input_channels(input)
        input_channels = self.prelu(input_channels)
        input_channels = self.norm(input_channels)
        if self.skip:
            res, skip = self.depthwise(input_channels)
            return res, skip

        res = self.depthwise(input_channels, skip=self.skip)
        return res

class DepthwiseConv(layers.Layer):
    def __init__(self, H, B, P, dilation, casual, skip=True):
        super(DepthwiseConv, self).__init__(name='DepthwiseConv')
        self.skip = skip
        if casual:
            padding_type = "casual"
            self.norm = cLN(H)
        else:
            padding_type = "same"
            self.norm = gLN(H)

        self.conv1d = layers.Conv1D(H, P, dilation_rate=dilation, padding=padding_type, groups=H)
        self.prelu = layers.PReLU()
        
        self.res_out = layers.Conv1D(B, 1)
        if self.skip:
            self.skip_out = layers.Conv1D(B, 1)
    
    def call(self, input):
        input = self.conv1d(input) 
        input = self.prelu(input)
        input = self.norm(input)
        res = self.res_out(input)
        if self.skip:
            skip_out = self.skip_out(input)
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
        skip_list = []
        for block in self.blocks:    
                if self.skip:
                    res, skip = block(block_input)
                    block_input = self.res_add([block_input, res])
                    skip_list.append(skip)
                else:
                    res = block(block_input)
                    block_input = self.res_add([block_input, res])
        if self.skip:
            skip_conn = self.skip_add(skip_list)
            return block_input, skip_conn
        return block_input
        
class cLN(layers.Layer):
    def __init__(self, H, EPS = 1e-8):
        super(cLN, self).__init__(name="cLN")
        self.EPS = EPS
        shape = (1,1,H)
        self.gamma = tf.Variable(tf.ones(shape),trainable=True,shape=shape)
        self.beta = tf.Variable(tf.zeros(shape),trainable=True,shape=shape)

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
        self.gamma = tf.Variable(tf.ones(shape),trainable=True,shape=shape)
        self.beta = tf.Variable(tf.zeros(shape),trainable=True,shape=shape)

    def call(self, input):
        # (M, K, H)
        E_f = tf.math.reduce_mean(input, axis=(1,2), keepdims=True) # (M, 1, 1)
        var = tf.math.reduce_variance(input, axis=(1,2), keepdims=True)# (M, 1, 1)

        normalized = self.gamma * (input - E_f) / tf.math.sqrt(var + self.EPS) + self.beta
        # (M, K, H)
        return normalized
