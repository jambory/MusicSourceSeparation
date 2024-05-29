import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers

from parameters import Parameters
    
class Encoder(layers.Layer):
    """ 
    First step of model not counting, preprocessing data. Puts the inputs through a 1D Convolutional layer with N output channels. The encoder also includes a activation
    function of a ReLU layer.

    Args: (taken from Parameters object)
        N (int): Encoder output channel size
        win (int): Kernel size of encoder.
        overlap (int): Amount samples each audio chunk will overlap adjacent chunks by.

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
        C: (int) Amount of sources being estimated
        N: (int) Encoder output channel size
        B: (int) Bottleneck Conv1DBlock output size
        R: (int) Amount of repeats of the Temporal Convolution block
        X: (int) Amount of times Conv1DBlock is applied in a Temporal Convolution block
        H: (int) Conv1DBlock input size
        P: (int) Size of kernel in depthwise convolutions
        Sc: (int=128) Channel size of skip connection and residual outputs.
        casual: (bool) Bool to decide what type of normalization is applied.
        skip: (bool) Bool to decide if skip connections are used.
    
    """
    def __init__(self, param: Parameters):
        super(Separator, self).__init__(name='Separator')
        self.C = param.C
        self.N = param.N
        self.skip=param.skip
        if param.casual:
            self.normalization = cLN(self.N)
        else:
            self.normalization = gLN(self.N) 
    
        self.bottle_neck = layers.Conv1D(param.B,1)
        
        self.temporal_conv = [TemporalConv(param.X, param.H, param.Sc, param.P, param.casual, skip=self.skip) for r in range(param.R)]

        self.skip_add = layers.Add()
        self.prelu = layers.PReLU()

        self.m_i = layers.Conv1D(self.C*self.N, 1)

    def call(self, w): # (M, K, N)
        M, K, _ = w.shape
        normalized_w = self.normalization(w) # (M, K, N)
        output = self.bottle_neck(normalized_w) # (M, K, B)
        if self.skip:
            skip_conn = tf.zeros(output.shape)
        for block in self.temporal_conv:
            if self.skip:
                output, skips = block(output) # (M, K, B), (M, K, B)
                skip_conn = self.skip_add([skip_conn, skips]) # (M, K, B)
            else:
                output = block(output) # (M, K, B)
        
        if self.skip:
            output = skip_conn

        output = self.prelu(output)
        estimated_masks = self.m_i(output) # (M, K, C*N)
        estimated_masks = activations.sigmoid(tf.reshape(estimated_masks, (M, self.C, K, self.N))) # (M, C, K, N)
        return estimated_masks # (M, C, K, N)
    
class Decoder(layers.Layer):
    """ 
    This is layer is final part of the model. Taking the features obtained, from the Conv1D blocks masks are finally combined into with the
    original mix into the estimated sources.

    Args:
        C: (int) Amount of sources being estimated
        N: (int) Encoder output channel size
        win (int): Kernel size of encoder.
        overlap (int): Amount samples each audio chunk will overlap adjacent chunks by.

    """
    def __init__(self, param: Parameters):
        super(Decoder, self).__init__(name='Decoder')
        self.C = param.C
        self.N = param.N
        self.win = param.win
        self.overlap = param.overlap


        self.transpose_conv = layers.Conv1DTranspose(1,self.win,strides=param.overlap,use_bias=False)

    def call(self, w, m_i): # (M, K, N) (M, C, K, N)
        M,K,_, = w.shape

        w = tf.reshape(tf.repeat(w, 4, axis=0),m_i.shape) # (M, K, N) -> (M, C, K, N)
        output = w * m_i # (M, C, K, N) * (M, C, K, N) 
        output = self.transpose_conv(tf.reshape(output, (M*self.C, K, self.N))) # (M*C, T, 1) 

        output = tf.reshape(output, (M,self.C,-1)) # (M, C, T)
        return output  # (M, C, T)

class Conv1DBlock(layers.Layer):
    """ 
    Conv1D block as described in this paper: https://arxiv.org/pdf/1809.07454v3. The structure is to take its input, change the channels
    to H w/ a PReLu, then normalize, perform a Depthwise Convolution then finally add the features the residuals from the input to create a 
    skip connected feature set.

    Args: 
        H: (int) Conv1DBlock input size
        Sc: (int=128) Channel size of skip connection and residual outputs.
        P: (int) Size of kernel in depthwise convolutions
        dilation: (int) Power 2 int, to decide dilation factor of convolution.
        casual: (bool) Bool to decide what type of normalization is applied.
        skip: (bool) Bool to decide if skip connections are used.

    Returns:
        (res, skip) or res ()

    """
    def __init__(self, H, Sc, P, dilation, casual, skip = True):
        super(Conv1DBlock, self).__init__(name='Conv1DBlock')
        self.skip = skip
        self.input_channels = layers.Conv1D(H,1)
        self.prelu = layers.PReLU()
        if casual:
            self.norm = cLN(H)
        else:
            self.norm = gLN(H)
        
        self.depthwise = DepthwiseConv(H, Sc, P, dilation, casual, skip=self.skip)
        self.res_add = layers.Add()

    def call(self, input): # (M, K, B/Sc)
        input_channels = self.input_channels(input) # (M, K, H)
        input_channels = self.prelu(input_channels)
        input_channels = self.norm(input_channels)
        if self.skip:
            res, skip = self.depthwise(input_channels)  # (M, K, Sc),(M, K, Sc)
            res = self.res_add([res, input]) # (M, K, Sc)
            return res, skip 

        res = self.depthwise(input_channels, skip=self.skip) # (M, K, Sc)
        res = self.res_add([res, input]) # (M, K, Sc)
        return res # (M, K, Sc)

class DepthwiseConv(layers.Layer):
    """ 
    A depthwise convolutional layer. The intuition of using a depthwise convolution over a normal
    convolution layer is as follows. 
    
    Typically for a convolution what you end up creating is K x K x C kernel(s),
    where each kernel spans the basic kernel size amongst every input channel. The depthwise convolution makes a K x K x 1
    kernel specifically for each input channel. So the number of output channels generally should always be the same as the input
    channel number. 

    Then take this output and perform a 1x1 convolution (essentially a dense layer), and make the number of output neurons be
    your initially desired output channel size and you get your depthwise convolution.

    It has the benefit of sometimes requiring less parameters to train.

    Args:
        H: (int) Conv1DBlock input size
        Sc: (int=128) Channel size of skip connection and residual outputs.
        P: (int) Size of kernel in depthwise convolutions
        dilation: (int) Power 2 int, to decide dilation factor of convolution.
        casual: (bool) Bool to decide what type of normalization is applied.
        skip: (bool) Bool to decide if skip connections are used.
    """
    def __init__(self, H, Sc, P, dilation, casual, skip=True):
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
        
        self.res_out = layers.Dense(Sc)
        if self.skip:
            self.skip_out = layers.Dense(Sc)
    
    def call(self, input): # (M, K, H)
        output = self.conv1d(input) # (M, K, H)
        output = self.prelu(output)
        output = self.norm(output)
        res = self.res_out(input) # (M, K, Sc)
        if self.skip:
            skip_out = self.skip_out(output) # (M, K, Sc)
            return res, skip_out # (M, K, Sc), (M, K, Sc)
        else:
            return res # (M, K, Sc)

class TemporalConv(layers.Layer):
    """ 
    To downsample the input and get features, the model applies a Conv1DBlock X times. 
    The outputs of the blocks are added together to encompass the skip connection features.
    The process of repeating the Conv1DBlocks X times and collecting features is known as a
    TemporalConvolution. This process itself is also repeated R times which is shown in the Separator.

    Args: 
        X: (int) Amount of times Conv1DBlock is applied in a Temporal Convolution block
        H: (int) Conv1DBlock input size
        Sc: (int=128) Channel size of skip connection and residual outputs. Typically will be the same as B.
        P: (int) Size of kernel in depthwise convolutions
        casual: (bool) Bool to decide what type of normalization is applied.
        skip: (bool) Bool to decide if skip connections are used.
    """
    def __init__(self, X, H, Sc, P, casual, skip=True):
        super(TemporalConv,self).__init__(name='TemporalConv')
        self.skip=skip
        self.blocks = []
        for i in range(X):
            dilation = 2 ** i
            self.blocks += [Conv1DBlock(H, Sc, P, dilation, casual, skip=self.skip)]
        self.res_add = layers.Add()
        self.skip_add = layers.Add()
    
    def call(self, block_input): # (M, K, B)
        if self.skip:
            skip_conn = tf.zeros(block_input.shape) # Creates blank tensor for skip features to be added to. 
        for block in self.blocks:    
                if self.skip:
                    res, skip = block(block_input) # (M, K, Sc), (M, K, Sc)
                    block_input = self.res_add([block_input, res]) # (M, K, Sc)
                    skip_conn = self.skip_add([skip_conn,skip]) # (M, K, Sc)
                else:
                    res = block(block_input) # (M, K, Sc)
                    block_input = self.res_add([block_input, res]) # (M, K, Sc)
        if self.skip:
            return block_input, skip_conn # (M, K, Sc), (M, K, Sc)
        else:
            return block_input # (M, K, Sc)
        

class cLN(layers.Layer):
    """ 
    Casual normalization process. Generally for when it is determined normalization should not be dependent on future values.
    More information can be found here: https://arxiv.org/pdf/1809.07454v3.

    Args: 
        channel_output: (int) Input and output channel number. Generally just keeps the same dimension.
        EPS: (float) Error term
    """
    def __init__(self, channel_output, EPS = 1e-8):
        super(cLN, self).__init__(name="cLN")
        self.EPS = EPS
        shape = (1,1,channel_output)
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
    """ 
    Global normalization process. Generally for when it is determined normalization should can be dependent on future values.
    More information can be found here: https://arxiv.org/pdf/1809.07454v3.

    Args: 
        channel_output: (int) Input and output channel number. Generally just keeps the same dimension.
        EPS: (float) Error term
    """
    def __init__(self, channel_output, EPS = 1e-8):
        super(gLN, self).__init__(name="gLN")
        self.EPS = EPS
        shape = (1,1,channel_output)
        self.gamma = tf.Variable(tf.ones(shape),trainable=True,shape=shape,name="Global Gamma")
        self.beta = tf.Variable(tf.zeros(shape),trainable=True,shape=shape,name="Global Beta")

    def call(self, input):
        # (M, K, H)
        E_f = tf.math.reduce_mean(input, axis=(1,2), keepdims=True) # (M, 1, 1)
        var = tf.math.reduce_variance(input, axis=(1,2), keepdims=True)# (M, 1, 1)

        normalized = self.gamma * (input - E_f) / tf.math.sqrt(var + self.EPS) + self.beta
        # (M, K, H)
        return normalized
