# MusicSourceSeparation

*Wilcox, Coby*

A comprehensive look at music source separators, with implementations and explanations of the underlying ideas and algoithms behind them and full implementations of different separation models.

`main_report.ipynb` : A overall report of my methodology, results, intentions going in, and sources for my information.
`display.py` : Module for displaying results of project in 'main_report'.

*audio_processing* : An overview of how data is loaded and processed in a python environment. Begins with loading a .wav file and how the data itself appears, then looks at a numpy implementation of a short-time fourier transform pipeline, finally going over creating and interpretting a spectrogram. Also another report is included 'spectrogram_limits.ipynb' that goes over some of the issues of spectrogram-based models.

    - `audio_processing.py` : Module for all functions used in audio_processing.
    - `spectrogram_report.ipynb` : In depth report for how a spectrogram is created from the ground up.
    - `spectrogram_limits.ipynb` : A very basic, nonapplicable spectrogram based model is created to demonstrate, the general problem with only relying on spectrograms for a MSS model. 

*residual_analysis* : This is a deep dive into the where and the why of the errors in the spectrogram based MSS model, detailed in 'spectrogram_limits.ipynb'.

    - `residual.py` : Module for all functions used in residual_analysis.
    - `residual_analysis.ipynb` : In depth report for exactly where the errors in the 'spectrogram_limits' model are coming from.

*mss_basic_implementation* : A basic music source separation method involving median filtering. This was mainly to give myself an basic understanding of how a music source separation model would work before jumping into a neural network model.

    - `hps.py` : Module for all functions used in mss_basic_implementation.
    - `hps_model.ipynb` : A report going over the harmonic-percussive MSS model.

*neural_network_basics* : Numpy implementation of neural networks which greatly aided my understanding and implementation of the neural network based MSS model.

    - `nn_scripts/activations.py` : Module for all activation function layers used in the numpy neural network implementations.
    - `nn_scripts/dense.py` : Script for dense layers used in the numpy neural network implementations.
    - `nn_scripts/layer.py` : Script for base layer used in the numpy neural network implementations.
    - `nn_scripts/loss.py` : Module for all loss function layers used in the numpy neural network implementations.
    - `nn_scripts/network.py` : Script for for forward/back propogation used in the for loop neural network implementation.
    - `nn_basic.ipynb` : An implementation of the most basic for loop implemetation of a neural network.
    - `nn_report.ipynb` : A report going over the a slightly more complex numpy implemetation of a neural network, using custom objects.   
    - `cnn_report.ipynb` : A report going over convolutional neural networks, with a numpy implementation.

*mss_nn_implementation* : My attempt at recreating the model detailed in this paper: *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation*[1]. 

    - `convtas.py` : A module for the overall ConvTas-Net implementation I made in TensorFlow.
    - `parameters.py` : A script for the parameters object made for the ConvTas-Net model implementation.
    - `layer.py` : A module for the different layers I made to use within the ConvTas-Net model. 
    - `loss.py` : A module for the different loss functions I made to use within the ConvTas-Net model. 
    - `datasets.py` : A module for object classes used to create Tensor datasets for the model to be trained with. 
    - `training_final_full/` : A saved trained model that I created.
    - `convtas_report.ipynb` : A report going the architecture and intuition behind the ConvTas-Net model and my implemetation.
    - `model_report.ipynb` : A report the results of my model implementation.



    
