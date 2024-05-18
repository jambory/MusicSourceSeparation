from layer import Encoder, Separator, Decoder
from tensorflow.keras import models
from parameters import Parameters

# I heavily relied upon, upon Jane Wu's TensorFlow implementation (https://github.com/JaneWuNEU/Conv-TasNet-2) to get this to work.
# nplab's and kaituoxu's PyTorch implementations of the model were also instrumental for my understanding of the model.

class ConvTasNet(models.Model):
    """
    Full Conv-TasNet model, coded in TensorFlow. 
    
    To explain what the model is doing in the most basic way, it takes a dataset of overlapping chunks of a audio in a waveform format then 
    encodes them using a simple 1D convolution. With this input data, the model is then put through several iterations of downsampling to get 
    features from the data. These features are then turned into separate masks for each target source in the audio data. Finally the masks are 
    applied to the orginal encoded data, which is finally put back into its orginal waveform data format with a Transpose 1D convolution.

    To understand the shape of the data throughout the model, I highly recommend looking at the Parameters class documentation, but I tried to 
    give documentation to the important paramters throughout the layers documentation as well. 
    """
    def __init__(self, param: Parameters):
        super().__init__(name="ConvTasNet")
        self.param = param


        self.encoder = Encoder(self.param)
        self.separator = Separator(self.param)
        self.decoder = Decoder(self.param)

    def call(self, x):
        w = self.encoder(x)
        m_i = self.separator(w)

        decoded = self.decoder(m_i, w)

        return decoded