import numpy as np
import librosa
import musdb
import IPython.display as ipd
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import audio_processing as ap

def load_waltz():
    """ 
    Loads Clara Berry And Wooldog - Waltz For My Victims, from the MUSDB sample dataset. 

    Returns: 
        stems: (list) list of the audio data for each stem contained in the song. Transposed to be compatible with librosa.
        track: (musdb.TrackObject) the actual track info of the song, just incase any extra data is required for analysis
        sr: (int) sample rate of music clip, should be 44100.
    """
    mus = musdb.DB(download=True)
    track = mus[21]
    track.name

    sr = track.rate
    stems = [librosa.to_mono(audio.T) for audio in track.stems] # musdb parses data in transposed format to most audio objects.
    ipd.display(ipd.Audio(track.audio.T, rate=sr))

    return stems, track, sr


def orthogonal_projection(A, B):
    """
    Compute the orthogonal projection of vector A onto vector B.

    Parameters:
    A (array-like): Vector A
    B (array-like): Vector B

    Returns:
    numpy.ndarray: The projection of A onto B
    """
    A = np.array(A)
    B = np.array(B)
    
    # Compute the dot product of A and B
    dot_product_AB = np.dot(A, B)
    
    # Compute the dot product of B with itself
    dot_product_BB = np.dot(B, B)
    
    # Compute the projection scalar
    projection_scalar = dot_product_AB / dot_product_BB
    
    # Compute the projection of A onto B
    projection = projection_scalar * B
    
    return projection



