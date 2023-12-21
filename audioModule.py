import os
import warnings
warnings.filterwarnings('ignore')
import wave


import random
import librosa
import numpy as np
import soundfile as sf
# install pydub for using HighPassFilter and play
from pydub.playback import play
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
# import simpleaudio as sa
import matplotlib.pyplot as plt
#from helper import _plot_signal_and_augmented_signal
from IPython.display import Audio
import librosa.display as dsp
# import mir_eval
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchsummary import summary 
import os

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

from torch import nn
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor

import torchvision
# from tflite_model_maker import audio_classifier
import tensorflow as ts


plt.rcParams["axes.labelsize"] = 'medium'
plt.rcParams["axes.titlecolor"] = 'red'
plt.rcParams["axes.titlesize"] = 'large'
#plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 18

class audioPreprocessing:
    
    
    def __init__(self, sample_rate = None, num_samples = None, duration = None):
        self.signal = None
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.duration = duration
        
    def readAudio(self, fileName, dur = 60):
        self.signal, self.sample_rate = librosa.load(fileName, sr=22050)#, mono = isMono,duration = dur)
#         self.signal, self.sample_rate = torchaudio.load(fileName, normalize=True)
        self.num_samples = self.signal.shape[0]
        self.duration = librosa.get_duration(self.signal)

        return self.signal
    
    def add_white_noise(self, signal, noise_percentage_factor):
        noise = np.random.normal(0, signal.std(), signal.size)
        augmented_signal = signal + noise * noise_percentage_factor
        return augmented_signal


    def time_stretch(self, signal, time_stretch_rate):
        """Time stretching implemented with librosa:
        https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
        """
        return librosa.effects.time_stretch(signal, time_stretch_rate)


    def pitch_scale(self, signal, sr, num_semitones):
        """Pitch scaling implemented with librosa:
        https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
        """
        return librosa.effects.pitch_shift(signal, sr, num_semitones)


    def random_gain(self, signal, min_factor=0.1, max_factor=0.12):
        gain_rate = random.uniform(min_factor, max_factor)
        augmented_signal = signal * gain_rate
        return augmented_signal


    def invert_polarity(self, signal):
        return signal * -1
    

    def _resample_if_necessary(self, signal, targetSampleRate):
        if targetSampleRate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(self.sample_rate, targetSampleRate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_if_necessary(self, signal, targetNumSamples = None):
        if targetNumSamples is None:
            if signal.shape[1] > self.num_samples:
                signal = signal[:, :self.num_samples]
        else:
            if signal.shape[1] > targetNumSamples:
                signal = signal[:, :targetNumSamples]
            
        return signal

    def _right_pad_if_necessary(self, signal, targetNumSamples = None):
        length_signal = signal.shape[1] 
        if targetNumSamples is None:
            if length_signal < self.num_samples:
                num_missing_samples = self.num_samples - length_signal
                last_dim_padding = (0, num_missing_samples)
                signal = torch.nn.functional.pad(signal, last_dim_padding)
        else:
            if length_signal < targetNumSamples:
                num_missing_samples = targetNumSamples - length_signal            
                last_dim_padding = (0, num_missing_samples)
                signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    
    def audioAugmentation(self, signal):
        # Raw audio augmentation
        augment_raw_audio = Compose(
            [
                AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=1),
                PitchShift(min_semitones=-2, max_semitones=2, p=1),
                HighPassFilter(min_cutoff_freq=3000, max_cutoff_freq=4000, p=1)
            ]
        )
        augmented_signal = augment_raw_audio(signal, self.sample_rate)
        return augmented_signal

    def audioAugmentation1(self, signal, i):
        # Raw audio augmentation
        aug_transform = []
        aug_transform1 = [
                AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=1),
                PitchShift(min_semitones=-2, max_semitones=2, p=1),
                HighPassFilter(min_cutoff_freq=3000, max_cutoff_freq=4000, p=1)
            ]
        for j in i:
            aug_transform.append(aug_transform1[j])


        augment_raw_audio = Compose(aug_transform)
        augmented_signal = augment_raw_audio(signal, self.sample_rate)
        return augmented_signal

    
    def plotSpectrum(self, signal, sample_rate, plotType = 'linear', title = "", ax = None):
            if ax is None:
                fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9,5));
            
            d = librosa.stft(signal);
            D = librosa.amplitude_to_db(np.abs(d),ref=np.max);
            dsp.specshow(D, y_axis = plotType, x_axis ='s', sr=sample_rate, ax = ax);
            ax.set(title = title);
            ax.label_outer();
#             fig.colorbar(img, ax = ax, format='%+2.f dB')
        
    def plotBeforeAfter(self, signal, augmented, sample_rate):
        fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (15,10))
        librosa.display.waveshow(signal, sr = sample_rate,ax = ax[0])
        ax[0].axis('off')
        ax[0].set(title = 'Original Signal')
        
        librosa.display.waveshow(augmented, sr = sample_rate, ax = ax[1])
        ax[1].set(title = 'Augmented Signal')
        ax[1].axis('off')

        plt.show()
