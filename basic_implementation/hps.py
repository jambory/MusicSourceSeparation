import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display

def create_spectrogram(audio, ax, win_size=2048, hop=None, mus=True):
    """ 
    Creates spectrograms of a given audio signal. The function first applies a short-time Fourier Transform (using librosa), 
    then creates the power maginitude spectrogram by simply taking the FT signal and applying an absolute value function and 
    squaring the elements. Then to visualize the spectrogram it uses another premade function from the librosa package.

    Args:
    - audio (array-like) : an array of a input audio signal to be transformed.
    - ax (axes obj) : axes to plot spectrogram on.
    - win_size (int) : size of window for Fourier Transform, should be a power 2 int.
    - hop (int) : hop size of Fourier Transform, if not set defaults to win_size // 4.
    - mus (bool) : If set to True performs a small transformation to the audio signal to account for musDB audio file format.

    Returns:
    - img : plot of spectrogram.
    - S (array-like) : array of power magnitude spectrogram elements.
    - X (array-like) : array of stft elements.
    """
    if hop is None:
        hop = win_size // 4
    if mus:
        audio = librosa.to_mono(audio.T)
    
    X = librosa.stft(audio, n_fft=win_size, hop_length=hop)
    S = np.abs(X)**2

    img = librosa.display.specshow(librosa.amplitude_to_db(S), y_axis='log', x_axis='time',cmap='magma', ax=ax)

    return img, S, X

def plot_spectrograms(track):
    """ 
    Takes a stems audiofile from the musdb dataset, and returns a plot giving spectrograms for each stem. 

    Args:
    - track (musdb.audio_classes.MultiTrack) : Track to be plotted.
    Returns:
    - specs (array-like) : a list of power magnitude spectrogram values, each index correlating with a source of the track.
    - X_s (array-like) : a list of STFT transformed elements each index correlating to a source of the track.
    """
    stem_order = ['Mix', 'Drums', 'Bass', 'Other', 'Vocals']

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    specs = []
    X_s = []

    for i,stem in enumerate(track.stems):
        img, S, X = create_spectrogram(stem, ax=axes[i])
        X_s.append(X)
        title = stem_order[i]
        if i > 0:
            axes[i].set_yticks([])
        axes[i].set_title(title)
        specs.append(S)
        
    
    fig.colorbar(img, format="%+2.0f dB")
    return specs, X_s

def median_filter_horizontal(x, filter_len):
    """Apply median filter in horizontal direction

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        x (np.ndarray): Input matrix
        filter_len (int): Filter length

    Returns:
        x_h (np.ndarray): Filtered matrix
    """
    return signal.medfilt(x, [1, filter_len])

def median_filter_vertical(x, filter_len):
    """Apply median filter in vertical direction

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        x: Input matrix
        filter_len (int): Filter length

    Returns:
        x_p (np.ndarray): Filtered matrix
    """
    return signal.medfilt(x, [filter_len, 1])

def med_filter_specs(specs, f_lens, hori=True):
    """ 
    Returns the arrays and plots the spectrograms of median filtered, spectrogram matrices. Allows you to specify if you want
    horizontal or vertical filters.

    Args:
        specs (array-like) : list of spectrogram values, for each source you want to plot.
        f_lens (int) : the length of the filter to be applied, normally this would be a 2 dimensional shape, but in this case
        we only want a kernel in the shape of a line.
        hori (bool) : Indicates whether you want a horizontal median filter or vertical line filter.
    Returns:
        filtered_audio : stft audio mask data, array of median values of spectrogram values.
    """
    filtered_audio = []
    titles = ["Mix", "Drums", "Vocals"]
    fig, axes = plt.subplots(1,3, figsize=(15,4)) 
    for i,source in enumerate(specs):
        if hori:
            med_filter = median_filter_horizontal(source, f_lens)
        else:
            med_filter = median_filter_vertical(source, f_lens)
        filtered_audio.append(med_filter)
        img = librosa.display.specshow(librosa.amplitude_to_db(med_filter), y_axis='log', x_axis='time',cmap='magma', ax=axes[i])
        axes[i].set_title(titles[i])
    fig.colorbar(img, format="%+2.0f dB")
    return filtered_audio

def plot_binary_masks(h_filtered_signals, v_filtered_signals):
    """ 
    Calculates and plots the binary masks.

    Args: 
        h_filtered_signals (array-like) : stft audio mask data for harmonic data.
        v_filtered_signals (array-like) : stft audio mask data for percussive data.
    Returns:
        M_bin_h, M_bin_p : Mask data for harmonic and percussive sources
    """
    fig, ax = plt.subplots(1,2, figsize=(8,3)) 

    M_bin_h = np.int8(h_filtered_signals[0] >= v_filtered_signals[0])
    M_bin_p = np.int8(h_filtered_signals[0] < v_filtered_signals[0])
    img = librosa.display.specshow(M_bin_h, x_axis='time', ax=ax[0])
    img = librosa.display.specshow(M_bin_p, x_axis='time', ax=ax[1])
    ax[0].set_title("Binary Harmonic Mask of Mix")
    ax[1].set_title("Binary Percussive Mask of Mix")

    return M_bin_h, M_bin_p