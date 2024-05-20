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

def make_mask(stems):
    """ 
    Creates and plots masks using method shown in spectrogram_limits.ipynb.
    """
    fig1, axes1 = plt.subplots(1,5, figsize=(15,4))
    fig2, axes2 = plt.subplots(1,5, figsize=(15,4))

    spec_data = [ap.create_spectrogram(stem, ax=axes1[i]) for i, stem in enumerate(stems)]
    masks = 




