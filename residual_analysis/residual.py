import numpy as np
import librosa
import musdb
import IPython.display as ipd
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import audio_processing as ap

sr=44100

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


def make_masks(stems):
    """ 
    Creates and plots masks using method shown in spectrogram_limits.ipynb.

    Args: 
        stems: (list) list of audio data of mix and target sources.

    Returns:
        masks: (list) list of masks generated. Order returned is ('Mix','Drums','Bass','Other','Vocals').
    """
    fig1, axes1 = plt.subplots(1,5, figsize=(15,4))
    fig2, axes2 = plt.subplots(1,5, figsize=(15,4))
    sources = ['Mix','Drums','Bass','Other','Vocals'] # list for labels

    spec_data = [ap.create_spectrogram(stem, ax=axes1[i], title=f'Target: {sources[i]}', hop=1024) for i, stem in enumerate(stems)] # creates list of spectrogram data
    
    masks = [(spec_data[i][1] / np.maximum(spec_data[i][1], spec_data[0][1]) + 1e-8) for i in range(len(spec_data))] # makes mask

    for i,mask in enumerate(masks):
        img =librosa.display.specshow(librosa.amplitude_to_db(mask), y_axis='log', x_axis='time',cmap='coolwarm_r', ax=axes2[i])
        axes2[i].set_title(f'Mask: {sources[i]}')
        if i !=0:
            axes1[i].set_xlabel('')
            axes1[i].set_ylabel('')
            axes2[i].set_xlabel('')
            axes2[i].set_ylabel('')
    
    fig1.colorbar(spec_data[4][0], format="%+2.0f dB")
    fig2.colorbar(img, format="%+2.0f dB")
        
    return spec_data, masks

def apply_masks(spec_data, masks):
    """ 
    Applies generated masks, to given spectrogram mix and displays the.

    Args: 
        spec_data: (list) list of spectral data.
        masks: (list) list of masks.
    Returns:
        est_sources: (list) list of applied masks to spectral mix.
    """
    fig1, axes1 = plt.subplots(1,4, figsize=(12,4))
    sources = ['Mix','Drums','Bass','Other','Vocals']
    est_sources = []
    for i in range(0,4):
        est_source = spec_data[0][2] * masks[i+1]
        est_sources.append(est_source)
        img1=librosa.display.specshow(librosa.amplitude_to_db(np.abs(est_source)), y_axis='log', x_axis='time',cmap='magma', ax=axes1[i])
        axes1[i].set_title(f'Estimated source: {sources[i+1]}')
        axes1[i].set_xlabel('') # Removes labels for visual clarity
        axes1[i].set_ylabel('')

    fig1.colorbar(img1, format="%+2.0f dB")

    return est_sources

def plot_residuals_spec(spec_data, s_hat):
    """ 
    Measures and plots residuals on spectrograms.
    Args:
        spec_data (list) list of spectral data
    Returns:
        s_hat (list) list of source estimations.
    """
    fig, axes = plt.subplots(1,4, figsize=(15,4))
    residuals = []
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        res = (spec_data[i+1][1] - np.abs(s_hat[i]))**2 # performs residual calculation
        residuals.append(res)
        img = librosa.display.specshow(res, y_axis='log', x_axis='time',cmap='Reds_r', ax=axes[i])
        axes[i].set_title(f'Residuals: {sources[i]}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    # Creates custom ticks
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', spacing='proportional')
    cbar_ticks = [np.min(res),np.max(res)]
    cbar.set_ticks(cbar_ticks)

    # Custom tick labels
    labels = ["No Diff", "Large Diff"]
    cbar.set_ticklabels(labels)

    return residuals

def playback_sources(spec_data, s_hat):
    """ 
    Takes stft original and masked data and applies istft's to them each along with displaying each audio.

    Args:
        spec_data (list): list of img, magnitude stft arrays and orginal stft arrays respectively
        s_hat (list): list of masked stft data
    Returns:
        est_signal (list): list of estimated audio source signals
        true_signals (list): list of actual audio source signals
    """

    est_signal = [librosa.istft(est_source, hop_length=1024) for est_source in s_hat]
    true_signal = [librosa.istft(spec_data[i][2], hop_length=1024) for i in range(1,5)]

    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        print('True Signal:', sources[i])
        ipd.display(ipd.Audio(true_signal[i], rate=sr))
        print('Estimated Signal:', sources[i])
        ipd.display(ipd.Audio(est_signal[i], rate=sr))
        print()
    
    return est_signal, true_signal

def plot_audio_signals(est_signals, true_signals):
    """ 
    Plots audio signals of the true and estimated sources. Overlays them on top of each other.
    Args:
        est_signal (list): list of estimated audio source signals
        true_signals (list): list of actual audio source signals
    """
    fig, axes = plt.subplots(1,4,figsize=(12,4))
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        axes[i].plot(est_signals[i], label="Estimated Source", alpha=.4)
        axes[i].plot(true_signals[i], label="True Source", alpha=.4, color='red')
        axes[i].legend()
        axes[i].set_title(f'{sources[i]}')

def display_csd(est_signals, true_signals):
    """ 
    Calculates and displays cosprectrum graphs for each source with their estimated and actual signals.
    Args:
        est_signal (list): list of estimated audio source signals
        true_signals (list): list of actual audio source signals
    """
    fig, axes = plt.subplots(1,4,figsize=(12,4))
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        axes[i].cohere(est_signals[i], true_signals[i], Fs = 44100, NFFT=2048, scale_by_freq=True, noverlap=1024)
        axes[i].set_title(f'{sources[i]}')

def plot_residual_signals(est_signals, true_signals):
    """ 
    Displays the actual residuals of the estimated vs. actual source signals.
    Args:
        est_signal (list): list of estimated audio source signals
        true_signals (list): list of actual audio source signals
    Returns:
        res_signals (list): list of residuals for each source signal.
    """
    res_signal = [est_signals[i]-true_signals[i] for i in range(4)]
    fig, axes = plt.subplots(1,4,figsize=(12,4))
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        axes[i].plot(res_signal[i])
        axes[i].set_title(f'{sources[i]}')

    return res_signal

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_res_acf(est_signals, true_signals):
    """ 
    Calculates and displays acf graphs for each source with their estimated signals.
    Args:
        est_signal (list): list of estimated audio source signals
        true_signals (list): list of actual audio source signals
    """
    res_signal = [est_signals[i]-true_signals[i] for i in range(4)]
    fig, axes = plt.subplots(1,4,figsize=(12,4))
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        plot_acf(res_signal[i],ax=axes[i],alpha=.05)
        axes[i].set_title(f'{sources[i]}')

def plot_res_pacf(est_signals, true_signals):
    """ 
    Calculates and displays pacf graphs for each source with their estimated signals.
    Args:
        est_signal (list): list of estimated audio source signals
        true_signals (list): list of actual audio source signals
    """
    res_s
    res_signal = [est_signals[i]-true_signals[i] for i in range(4)]
    fig, axes = plt.subplots(1,4,figsize=(12,4))
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        plot_pacf(res_signal[i],ax=axes[i],alpha=.05,)
        axes[i].set_title(f'{sources[i]}')

def get_phase(spec_data, s_hat):
    """ 
    Displays phase for each target source along with the full mix. Important to note that for the estimated sources,
    they all have the mix phase.
    Args:
        spec_data (list): list of img, magnitude stft arrays and orginal stft arrays respectively
        s_hat (list): list of masked stft data
    Returns:
        phase_spec (list): list of phase information for targets and mix
        phase_est (list): list of phase information for estimated sources. They are all the same, this is just for demonstration purposes

    """
    phase_spec = [np.angle(spec_data[i][2]) for i in range(5)]
    phase_est = [np.angle(spectrogram) for spectrogram in s_hat]

    fig1, axes1 = plt.subplots(1,5, figsize=(15,4))
    sources = ['Mix','Drums','Bass','Other','Vocals']
    for i in range(5):
        img1=librosa.display.specshow(librosa.amplitude_to_db(phase_spec[i]), y_axis='log', x_axis='time',cmap='magma', ax=axes1[i]) # Plots spectrograms
        axes1[i].set_title(f'Target: {sources[i]}')
        axes1[i].set_xlabel('') # Removes labels for visual clarity
        axes1[i].set_ylabel('')

    fig1.colorbar(img1, format="%+2.0f")
    return phase_spec, phase_est

def calculate_phase_error(original_phase, masked_phase):
    """ 
    Calculates the phase error of the estimated phase and the target source phases.
    Small explanation of why phase error is set up this way:

    Since phase values can only value from $-\pi$ to $\pi$, the value should wrap around if the value exceeds that. Just subtracting the two values wouldn't immediately do that, 
    but taking the values and putting through this function $f(x) = e^{-ix}$ effectivelu plots them on a circle allowing the values to stay within their range.

    Args:
        original_phase (list): list of phase information for targets
        masked_phase (list): list of phase information for estimated sources
    """
    phase_error = []
    for i in range(4):
        phase_error.append(np.angle(np.exp(1j * (original_phase[i] - masked_phase[i])))) # the reason that the residual is put through this formula is to deal with the cyclic nature of phase.
    return phase_error

def plot_phase_error(phase_error, ticks=True):
    fig, axes = plt.subplots(1,4, figsize=(15,4))
    sources = ['Drums','Bass','Other','Vocals']

    for i in range(4):
        img = librosa.display.specshow(np.abs(phase_error[i]), y_axis='log', x_axis='time',cmap='autumn', ax=axes[i])
        axes[i].set_title(f'Residuals: {sources[i]}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    # Creates custom ticks
    if ticks:
        cbar = fig.colorbar(img, ax=axes, orientation='vertical', spacing='uniform')
        cbar_ticks = [0,np.max(phase_error[3])]
        cbar.set_ticks(cbar_ticks)

        # Custom tick labels
        labels = [ "No Diff", "Pi"]
        cbar.set_ticklabels(labels)

def create_zero_trans():    
    from matplotlib.colors import LinearSegmentedColormap

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('Reds_r')(range(ncolors))

    alpha_vals = [0]
    alpha_vals.extend([1]*(ncolors-1))
    # change alpha values
    color_array[:,-1] = alpha_vals

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='red_zero_trans',colors=color_array)

    # register this new colormap with matplotlib
    try:
        plt.colormaps.register(cmap=map_object)
    except:
        pass

def plot_res_over_phase(phase_error, residuals, ticks=True):
    fig, axes = plt.subplots(1,4, figsize=(16,4))
    sources = ['Drums','Bass','Other','Vocals']
    try:
        create_zero_trans()
    except:
        pass

    for i in range(4):
        img1 = librosa.display.specshow(np.abs(phase_error[i]), y_axis='log', x_axis='time',cmap='Blues', ax=axes[i])
        img = librosa.display.specshow(residuals[i], y_axis='log', x_axis='time',cmap='red_zero_trans', ax=axes[i])
        axes[i].set_title(f'Residuals: {sources[i]}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    # Creates custom ticks
    if ticks:
        cbar = fig.colorbar(img1, ax=axes, orientation='vertical', spacing='uniform')
        cbar_ticks = [0,np.max(np.abs(phase_error[3]))]
        cbar.set_ticks(cbar_ticks)

        # Custom tick labels
        labels = [ "No Diff", "Pi"]
        cbar.set_ticklabels(labels)

def display_interactive_plot_res(residuals, phase_error):
    import ipywidgets as widgets
    from IPython.display import display

    # Creates custom color map
    create_zero_trans()

    fig, axes = plt.subplots(1,4, figsize=(16,4))
    sources = ['Drums','Bass','Other','Vocals']


    for i in range(4):
        img = librosa.display.specshow(np.abs(phase_error[i]), y_axis='log', x_axis='time',cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Residuals: {sources[i]}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_ylim(0,512)
        

    img1 = librosa.display.specshow(residuals[0], y_axis='log', x_axis='time',cmap='inf_zero_trans', ax=axes[0])
    img2 = librosa.display.specshow(residuals[1], y_axis='log', x_axis='time',cmap='inf_zero_trans', ax=axes[1])
    img3 = librosa.display.specshow(residuals[2], y_axis='log', x_axis='time',cmap='inf_zero_trans', ax=axes[2])
    img4 = librosa.display.specshow(residuals[3], y_axis='log', x_axis='time',cmap='inf_zero_trans', ax=axes[3])


    # Create a checkbox widget
    show_plot_checkbox = widgets.ToggleButton(
        value=False,
        description='Show Residuals',
    )

    # Update function to be called when the slider's value changes
    def update_plot_visibility(change):
        img1.set_visible(change['new'])
        img2.set_visible(change['new'])
        img3.set_visible(change['new'])
        img4.set_visible(change['new'])
        fig.canvas.draw_idle()

    # Connect the checkbox to the update function
    show_plot_checkbox.observe(update_plot_visibility, names='value')

    # Display the checkbox
    display(show_plot_checkbox)

    # Display the plot
    plt.show()
