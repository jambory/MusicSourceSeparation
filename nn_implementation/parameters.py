

class Parameters:
    """ 
    Class of params that can be called into any other layer or part of the model to be called without having to be declared each time
    in its initialization.
    Args:
        C: (int=4) Amount of sources being estimated
        N: (int=512) Encoder output size
        B: (int=128) Conv1DBlock output size
        H: (int=512) Conv1DBlock input size
        P: (int=3) Kernel size of Conv1DBlocks
        Sc: (int=128) Channel size of skip connection and residual outputs.
        R: (int=3) Amount of repeats of the Temporal Convolution block
        X: (int=8) Amount of times Conv1DBlock is applied in a Temporal Convolution block
        win: (int=16) Size of encoder-decoder kernels
        overlap: (int=8) Amount of overlapping samples in adjacent chunks, typically L // 2
        skip: (bool=True) Decides if skips are included to create features for mask generation
        casual: (bool=False) Decides if to use casual normalization or global normalization
    """

    def __init__(self,
                #  M: int=200, # Batch Size
                #  K: int=40, # Amount of overlapping chunks in a single element of a batch
                #  L: int=16, # Initial encoder window dimension
                #  T: int=3000, # Number of samples in input audio signal.
                 C: int=4, # Amount of sources being estimated
                 N: int=512, # Encoder output size
                 B: int=128, # Conv1DBlock output size
                 H: int=512, # Conv1DBlock input size
                 P: int=3, # Kernel size of Conv1DBlocks
                 Sc: int=128, # Channel size of skip connection and residual outputs.
                 R: int=3, # Amount of repeats of the Temporal Convolution block
                 X: int=8, # Amount of times Conv1DBlock is applied in a Temporal Convolution block
                 win: int=16, # Size of encoder-decoder kernels
                 overlap: int=8, # Amount of overlapping samples in adjacent chunks, typically L // 2
                 skip: bool=True, # Decides if skips are included to create features for mask generation
                 casual: bool=False, # Decides if to use casual normalization or global normalization
                 ):
        self.C = C
        self.N = N
        self.B = B
        self.H = H
        self.R = R
        self.X = X
        self.P = P
        self.Sc = Sc
        self.win = win
        self.overlap = overlap
        self.skip = skip
        self.casual = casual


    def get_config(self) -> dict:
        return {"C": self.C,
                "N": self.N,
                "B": self.B,
                "H": self.H,
                "R": self.R,
                "X": self.X,
                "P": self.P,
                "Sc": self.Sc, 
                "win": self.win,
                "overlap": self.overlap,
                "skip": self.skip,
                "casual": self.casual}