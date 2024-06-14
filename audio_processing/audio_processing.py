import soundfile as sf
import matplotlib.pyplot as plt 
import numpy as np
import librosa
from IPython.display import Audio

def load_waveform(path, stereo=True):
    """ 
    Uses PySoundFile to read audio data from .wav file and combines audio channels into one using average.
    Spectrograms for source separation purposes will only make one spectrogram of the combined audio channels, 
    but for different purposes sometimes distinct spectrograms are made for each audio channel.

    Args:
    - path: file path for .wav file to be read.
    Returns:
    - data: Average of combined stereo channels waveform data or just return waveform data if stereo set to False.
    - samplerate: samplerate of waveform data.
    """
    
    data,samplerate = sf.read(path) # 'wav/claire_de_lune.wav'
    if stereo:
        data = np.mean(data, axis=1)

    return data, samplerate

def display_waveform(data, samplerate = None, ax=None, title="Waveform data for Claire de Lune"):
    """
    Takes in audio waveform data and visualizes it.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(data, alpha=0.8, )
    if samplerate:
        ax.set_xticks(np.arange(0,len(data), step=samplerate*30), np.arange(0,len(data)/samplerate, step=30))
        ax.set_xlabel("Time Seconds")
    else:
        ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

def split_wave(data, window_size=2048, hop_length=1024):
    """
    Split numpy representation of audio file into overlapping chunks.
    If the last chunks are not the correct window size, the function also pads the chunks with zeroes.

    Args:
    - data: audio signal data.
    - window_size (int) : size of chunks to split data.
    - hop_length (int) : size of jumps between each window; for audio processing you want overlapping chunks, so recommended size is
    window_size/2
    Returns:
    - frames: overlapping chunks of audio signal.
    """
    frames = []
    for start in range(0,(len(data)-window_size), hop_length):
        frames.append(data[start:start+window_size])
    # Since method of splitting can result in the last two chunks being smaller than window length, this checks will pad last two
    # windows with zeros if either are less than window_size
    return frames

def hamming_window(signal_length):
    """
    Generate a Hamming window of given length.

    Args:
    signal_length (int): Length of the window.

    Returns:
    - np.ndarray: Hamming window.
    """
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(signal_length) / (signal_length - 1.0))

def apply_window(signal, window_length=2048):
    """
    Apply a window to a signal.

    Args:
    signal (np.ndarray): Input signal.
    window (np.ndarray): Window function.

    Returns:
    np.ndarray: Windowed signal.
    """
    window = hamming_window(window_length)
    return np.multiply(signal, window)

def window_chunks(chunks):
    return list(map(apply_window, chunks))

def window_signal(data):
    """ 
    Takes audio signal data, splits into overlapping chunks, applies Hamming windowing, and visualizes changes.

    Args:
    - data : audio signal data

    Returns:
    - window_chunks (list) : list of np.ndarrays each with windowed chunk data.
    """
    window_size = 2048
    hop_length = 1024

    chunks = split_wave(data, window_size, hop_length)
    windowed_chunks = window_chunks(chunks)

    # Scaled version of hamming window to better visualize how the windowing effects the chunks.
    hamming = hamming_window(window_size) * 0.5

    fig1, axes1 = plt.subplots(1,5,figsize=(15,3))
    fig2, axes2 = plt.subplots(1,5,figsize=(15,3))
    fig1.suptitle("Non-Windowed Chunks")
    fig2.suptitle("Windowed Chunks")

    for i in range(5):
        axes1[i].plot(chunks[i+4950])
        axes1[i].plot(hamming, 'r--', alpha=0.3)
        axes1[i].set_yticks([])

        axes2[i].plot(windowed_chunks[i+4950])
        axes2[i].plot(hamming, 'r--', alpha=0.3)
        axes2[i].set_yticks([])

    return windowed_chunks

def fourier_transform(data):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal.

    Args:
    data (np.ndarray): Input audio signal.

    Returns:
    np.ndarray: DFT of the input signal.
    """
    N = len(data)
    n = np.arange(N)
    k = n.reshape((N, 1))
    omega = np.exp(-2j * np.pi * k * n / N)
    return np.dot(omega, data)

def stft_chunks(chunks, samplerate=44100):
    """ 
    Applies short time fourier transform to all chunks and plots 3, to demonstrate ideas
    """
    chunks = np.array(chunks)
    K, N = chunks.shape
    n = np.arange(N)
    k = n.reshape((N, 1))
    omega = np.exp(2j * np.pi * k * n / N)

    ft_chunks = chunks @ omega
    
    frequency_bins = np.fft.fftfreq(N, d=1/samplerate)

    fig, axes = plt.subplots(1,3, figsize=(9,4))
    fig.suptitle("Fourier Transformed Chunks")

    for i in range(3):
        axes[i].plot(np.abs(chunks[i+4950]))
        axes[i].set_xlim(left=0, right=2000)
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Magnitude')
        axes[i].grid(True)

    return ft_chunks, frequency_bins

def plot_spectrogram(data, sr=44100):
    from matplotlib.colors import LogNorm
    N = len(data[0])
    frequency_bins = np.fft.fftfreq(N, d=1/sr)
    indexes = np.arange(0,N//2-1, (N//2-1) // 10)
    freq_vals = []

    for i in indexes:
        freq_vals.append(int(frequency_bins[i]))

    power_spectrum = np.abs(data.T)**2

    # Plot each chunk as a separate spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(power_spectrum[:1023,:] + 1e-8),aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='Magnitude (dB)', format='%+2.0f dB')  # Add colorbar with label


    plt.yticks([])
    for i in range(len(indexes)):
        plt.annotate(freq_vals[i], xy=(0, indexes[i]), xytext=(-32, 0), textcoords='offset points', ha='left', va='center')

    plt.xlim(0,12000)
    plt.xlabel('Frame')
    plt.title('Spectrogram')
    plt.show()

def plot_song_sources(track):
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    track = [librosa.to_mono(audio.T) for audio in track.stems]
    types = ["mix", "drums", "bass", "other", "vocals"]
    for i,stem in enumerate(track):
        if i == 0:
            axes[0].plot(stem, label = types[i])
        else:
            axes[1].plot(stem, label = types[i], alpha=.6)


    axes[0].legend(loc = 'upper left')
    axes[1].legend(loc = 'upper left')

def create_spectrogram(audio, ax, win_size=2048, hop=None, title=None, display=True):
    """ 
    Creates spectrograms of a given audio signal. The function first applies a short-time Fourier Transform (using librosa), 
    then creates the power maginitude spectrogram by simply taking the FT signal and applying an absolute value function and 
    squaring the elements. Then to visualize the spectrogram it uses another premade function from the librosa package.

    Args:
    - audio (array-like) : an array of a input audio signal to be transformed.
    - ax (axes obj) : axes to plot spectrogram on.
    - win_size (int) : size of window for Fourier Transform, should be a power 2 int.
    - hop (int) : hop size of Fourier Transform, if not set defaults to win_size // 4.
    - title (str) : title for spectrogram plot.

    Returns:
    - img : plot of spectrogram.
    - S (array-like) : array of power magnitude spectrogram elements.
    - X (array-like) : array of stft elements.
    """
    if hop is None:
        hop = win_size // 4

    X = librosa.stft(audio, n_fft=win_size, hop_length=hop)
    S = np.abs(X)

    if not display:
        return S,X

    img = librosa.display.specshow(librosa.amplitude_to_db(S), y_axis='log', x_axis='time',cmap='magma', ax=ax)
    if title:
        ax.set_title(title)

    return img, S, X

# def inverse_window(signal, window_length=2048):
#     """
#     Apply a window to a signal.

#     Args:
#     signal (np.ndarray): Input signal.
#     window (np.ndarray): Window function.

#     Returns:
#     np.ndarray: Windowed signal.
#     """
#     window = hamming_window(window_length)
#     return np.divide(signal, window)

# def inverse_window_chunks(chunks):
#     return list(map(inverse_window, chunks))

# def apply_inverse_ft(data):
#     K, N = data.shape
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     omega = np.exp(2j * np.pi * k * n / N)
#     return data @ omega

# def inverse_hamming(data, window_length=2048):
#     window = hamming_window(window_length)
#     inverse_window = lambda x: np.divide, window
#     return map(data, inverse_window)

# def combine_chunks(data, window_size=2048, hop_length=1024):
    wav = []
    for i,chunk in enumerate(data):
        step = (window_size-hop_length)*i
        wav.extend(chunk[step:step+(window_size-hop_length)])

    return wav


