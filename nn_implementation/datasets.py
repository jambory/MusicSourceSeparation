import tensorflow as tf
import numpy as np
import musdb
import random
import gc
from tqdm import tqdm
from typing import Union, List, Tuple, Dict

class DatasetParam:
    """ 
    Gives parameters for the dataset.
    Attributes:
        num_songs: Total number of songs.
        num_samples: Total number of samples in one batch.
        num_fragments: Total number of fragments in one sample. (fragments are just strings of bytes.)
        len_fragment: The length of each fragment object.
        overlap: Number of samples in which each adjacent pair of fragments overlap.
        repeat: Number of repeats.

    Note: when the author says `fragments` he is referring to selection of samples from a song
    whereas, the `samples` of a song refer to him taking a portion or `sample` of a song to enter into the NN

    TLDR: `Samples` in this section do not refer to samples the unit measurement within audio recording.
    """
    
    __slots__ = 'num_songs', 'num_samples', 'num_fragments', 'len_fragment', 'overlap', 'repeat'

    def __init__(self,
                num_songs: int= 100,
                num_samples: int=100,
                num_fragments: int=40,
                len_fragment: int=16,
                overlap: int = 8,
                repeat: int = 400):
        if overlap >= len_fragment:
            raise ValueError("overlap must be smaller than len_fragment.")
        
        self.num_songs = num_songs
        self.num_samples = num_samples
        self.num_fragments = num_fragments
        self.len_fragment = len_fragment
        self.overlap = overlap
        self.repeat = repeat

class DecodedTrack:
    """ 
    Contains decoded audio from the database.

    Attributes:
        length: Number of samples.
        mixed: A tuple of numpy arrays from the mixture.
        stems: Dictionary where the key is the name of the stem and value is tupe of the numpy arrays from the stem.
    """

    __slots__ = 'length', 'mixed', 'stems'

    @staticmethod # allows you to call method without first making an instance of the class.
    def from_track(track):
        mixed = (track.audio[:,0], track.audio[:,1]) # Changes from numpy array of shape (L, 2) to tuple of each audio channel data.
        length = mixed[0].shape[-1]
        stems = {}
        for stem in Dataset.STEMS: # Creating stems dict
            audio = track.targets[stem].audio
            stems[stem] = (audio[:,0], audio[:,1])
        return DecodedTrack(length, mixed, stems)
    
    def __init__(self, 
                 length: int,
                 mixed: Tuple[np.ndarray, np.ndarray],
                 stems: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        self.length = length
        self.mixed = mixed
        self.stems = stems

class Dataset:
    """ 
    Decodes audio from the database.
    Attributes:
        tracks: List of tracks.
        num_tracks: Number of tracks.
        decoded: List of decoded tracks.
        num_decoded: Number of decoded tracks in `decoded`
        max_decoded: Maximum number of decoded tracks.
        ord_decoded: The order in which each track is decoded.
        next_ord: The order which will be granted to the next decoded track

    """
    STEMS = "vocals", "drums", "bass", "other" # Used in DecodedTrack class for dict key generation

    def __init__(self, 
                 root: str, # path to root directory
                 subsets: Union[str, List[str]] = "train", # Allows for param to be either of the types in Union obj
                 max_decoded: int = 100):
        self.tracks = list(musdb.DB(root=root, subsets=subsets))
        self.num_tracks = len(self.tracks)
        self.decoded: Dict[str, Union[None, DecodedTrack]] = [None] * self.num_tracks
        self.num_decoded = 0
        self.max_decoded = max_decoded
        self.ord_decoded = [-1] * self.num_tracks # List of ints with length num_tracks 0 signaling first decoded track, and iterating after each decoding
        self.next_ord = 0 # What the heck do these do?

    def decode(self,
               indices: Union[int, List[int]]):
        if type(indices) == int:
            indices = [indices]
        if len(indices) > self.max_decoded:
            raise ValueError("Cannot decode more than max `max_decoded` tracks")
        
        indices = [idx for idx in indices if self.decoded[idx] == None]
        if indices:
            print(f"Decoding Audio {indices}...")
            for idx in tqdm(indices):
                self.ord_decoded # prints order of decoding
                if self.num_decoded == self.max_decoded: # I think this is a correction update to params to adjust to the max_decoded number
                    idx = np.argmin(self.ord_decoded)
                    self.decoded[idx] = None
                    self.num_decoded -= 1
                    self.ord_decoded[idx] -= 1
                    gc.collect() # cleans any variables not in use from memory
                self.decoded[idx] = DecodedTrack.from_track(self.tracks[idx])
                self.num_decoded += 1
                self.ord_decoded[idx] = self.next_ord
                self.next_ord += 1 # I assume this is to avoid any errors, running through indices, not really sure why it doesnt just break from the for loop

    def generate(self, 
                 p: DatasetParam):
        indices = list(range(self.num_tracks))
        random.shuffle(indices)
        indices = indices[:p.num_songs]
        self.decode(indices)

        duration = p.num_fragments * p.len_fragment - (p.num_fragments -1) * p.overlap

        # make `p.repeat` batches
        for _ in range(p.repeat): # create each batch
            x_batch = np.zeros(
                (p.num_samples * 2, p.num_fragments, p.len_fragment
            )) # use 2*num_samples as each fragment observation has a left and right value for each audio channel
            y_batch = np.zeros(
                (p.num_samples * 2, len(Dataset.STEMS), p.num_fragments, p.len_fragment)
            )

            # Make `2 * num_samples` samples for each batch
            for i in range(p.num_samples): # For each sample to be made
                track = self.decoded[random.choice(indices)] # take a random decoded track
                start = random.randint(0, track.length - duration) # take some random part to start the song at
            
                for j in range(p.num_fragments): # for each fragment in the sample
                    left = i * 2 # left is 2 times the index of the sample we are on
                    right = left+1 
                    begin = start + j * (p.len_fragment - p.overlap)
                    end = begin + p.len_fragment
                    x_batch[left][j] = track.mixed[0][begin:end]
                    x_batch[right][j] = track.mixed[1][begin:end]
                    
                    for c, stem in enumerate(Dataset.STEMS):
                        y_batch[left][c][j] = track.stems[stem][0][begin:end]
                        y_batch[right][c][j] = track.stems[stem][1][begin:end]

            yield x_batch, y_batch

    def make_dataset(self,
                     p: DatasetParam) -> tf.data.Dataset:
        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape(
                (p.num_samples * 2, p.num_fragments, p.len_fragment)
            ),
            tf.TensorShape(
                (p.num_samples * 2, len(Dataset.STEMS), p.num_fragments, p.len_fragment)
            ))
        return tf.data.Dataset.from_generator(lambda:self.generate(p),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
    
def load_unpack_tensordata():
    tf_data = tf.data.Dataset.load('tensors/tensor_dataset_small')
    x = []
    y = []
    for obs in tf_data:
        train.append(obs[0])
        test.append(obs[1])

    return x, y