import residual as res
import musdb
import numpy as np
import statsmodels.api as sm
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import sys
sys.path.append('model/')
import convtas, parameters
import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

mus = musdb.DB(download=True)

def load_track(mus, idx=21):
    """ 
    loads track from musdb sample dataset
    """
    track = mus[idx]
    audio = librosa.to_mono(track.audio.T)

    sr = track.rate
    stems = [librosa.to_mono(stem.T) for stem in track.stems] # musdb parses data in transposed format to most audio objects.

    return stems, audio, sr

def spec_calc_data():
    """ 
    Caclulates data used for calculations and visualizations in residual_analysis without any visualizations.
    Used in other display functions to get necessary data without haivng to calculate each step again.
    Returns:
        spec_data (list) each sources magnitude and fourier transform data.
        est_sources (list) each sources estimation for their target still in spectral format
        residuals (list) each sources estimation residuals for their target still in spectral format
        est_signals (list) each sources estimation for their target in audio format
        true_signals (list) each sources true target signal in audio format
        res_signals (list) each sources estimation residuals for their target in audio format
    """
    stems, _, sr = load_track(mus)
    spec_data, masks = res.make_masks(stems, display=False)
    est_sources = []
    residuals = []
    for i in range(0,4):
        est_source = spec_data[0][1] * masks[i+1]
        est_sources.append(est_source)
        residuals.append((spec_data[i+1][1] - np.abs(est_sources[i]))**2)

    est_signals = [librosa.istft(est_source, hop_length=1024) for est_source in est_sources]
    true_signals = [librosa.istft(spec_data[i][1], hop_length=1024) for i in range(1,5)]

    res_signals = [est_signals[i]-true_signals[i] for i in range(4)]
    return spec_data, est_sources, residuals, est_signals, true_signals, res_signals

def spectrogram():
    fig,ax = plt.subplots(1,1,figsize=(9,3))
    _, audio, _ = load_track(mus)
    spec_data, _, _, _, _, _ = spec_calc_data()
    img =librosa.display.specshow(librosa.amplitude_to_db(spec_data[0][0]), y_axis='log', x_axis='time',cmap='magma', ax=ax)
    fig.colorbar(img, format="%+2.0f dB")

def _sep_audio(output, output_size, pad, num_sources=4):
    separated=[]
    for i in range(num_sources):
        est = output[:,i,:]
        est_audio = np.reshape(est, (output_size,))[:-pad]
        separated.append(est_audio)
    
    return separated

def load_test_nn():
    mus_test = musdb.DB(download=True, subsets="test")
    stems, audio, sr = load_track(mus_test, 3)

    M, C, T = (len(audio)//400)+1, 4, 400
    pad = M*T - len(audio)
    padded_audio = np.pad(audio, (0,pad), mode='wrap') # Pads values at beginning to get correct size.
    model_input = np.reshape(padded_audio, (M, T))
    checkpoint_path = "trained_models/training_final_full/cp.ckpt"
    param = parameters.Parameters()
    cnn = convtas.ConvTasNet(param)
    cnn.load_weights(checkpoint_path)

    output = cnn(model_input)
    separated = _sep_audio(output, M*T, pad)

    return audio, stems, separated, cnn

def nn_plot_est_and_target(separated, original_audio, stems):
    # Creates structure for plots
    fig, axes = plt.subplots(1,4, figsize=(12,4))
    sources = ["Drums", "Bass", "Other", "Vocals"]

    for i in range(4):
        axes[i].plot(original_audio, label="Orignal Mix", alpha=.2)
        axes[i].plot(stems[i+1], label=f"Source True", alpha=.3,color='green')
        axes[i].plot(separated[i], label=f"Source Est", alpha=.5,color='orange')
        axes[i].grid()
        
        if i==0:
            fig.legend()

        axes[i].set_title(f"Source: {sources[i]}")
    
def display_nn_est():
    audio, stems, separated, _ = load_test_nn()
    nn_plot_est_and_target(separated, audio, stems)

def load_sarima(order=(3, 2, 7, 3)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    _, audio, _ = load_track(mus, 21)
    x, y = audio[:5000], audio[5000:5500]
    fig, axes = plt.subplots(1,4,figsize=(12,3))
    axes[0].plot(x)
    axes[0].set_title("Plot of Original Data")
    x1= np.diff(x,n=2)
    axes[1].plot(x1)
    axes[1].set_title("Plot of Diffed Data")
    plot_acf(x1, ax=axes[2])
    axes[2].set_title("ACF of Diffed Data")
    plot_pacf(x1, ax=axes[3])
    axes[3].set_title("PACF of Diffed Data")
    for i in range(3):
        axes[i].grid()
    
    model = SARIMAX(x, seasonal_order=order)

    model_fit = model.fit()
    return model_fit, y 

def diagnostics(model):
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    fig, ax = plt.subplots(2,2, figsize=(10,8))
    ax = ax.flatten()
    res = model.resid[1:]
    df = model.model_orders['ar'] + model.model_orders['ma']
    print(model.summary())

    ax[0].plot(res)
    ax[0].set_title("Residuals")
    plot_acf(res, ax=ax[1])
    ax[1].set_title("ACF Residuals")
    sm.qqplot(res,ax=ax[2])
    ax[2].set_title("Residuals QQ Plot")
    ljung = acorr_ljungbox(model.resid, lags=50, model_df=df)
    ax[3].stem(ljung["lb_pvalue"])
    ax[3].set_title("Residuals Q LjungBox Statistic P")

    for i in range(4):
        ax[i].grid()
    plt.show()

def forecast(model, y):
    fig, ax = plt.subplots(1,1,figsize=(10,3))
    forecast = model.forecast(steps=len(y))
    ax.plot(y, label="True Signal")
    ax.plot(forecast, label="Estimated Signal")
    ax.grid()
    ax.legend()
      
def listen_audio():

    sources = ["Drums", "Bass", "Other", "Vocals"]
    print('True Signal')
    for i in range(4):
        ipd.display(ipd.Audio(data=original_audio, rate=44100))
        print(f'Estimated {sources[i]}')
        ipd.display(ipd.Audio(data=separated[i], rate=44100))
        print(f'True {sources[i]}')
        ipd.display(ipd.Audio(data=stems[i], rate=44100))


def display_co_per(sr=44100):
    """ 
    Loads results of spectral analysis in concise format.
    Args:
        sr (int): sample rate. 
    """
    _, _, _, _, _, res_signals = spec_calc_data()
    psd, freq = res.periodogram_res(res_signals, sr=sr, display=False)
    smooth_psd, smooth_freq = res.norm_smooth_periodogram(psd, freq, window_size=1024, display=False)

    for i in range(4):   
        res.source_csd(res_signals, smooth_psd, smooth_freq, i)
