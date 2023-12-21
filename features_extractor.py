#!/usr/bin/env python
# coding: utf-8

# In[129]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation
import pandas as pd
import glob
from tqdm.auto import tqdm
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import librosa
import numpy as np
import os
def normalize(data):
    xmax, xmin =  data.max(), data.min()
    zi = 2 * ((data - xmin) / (xmax - xmin)) - 1
    return zi
    

def audio_mfcc(audio, n_mfcc = 50):
    signal, sr = librosa.load(audio, res_type = "kaiser_fast")
    mfcc_signal = np.mean(librosa.feature.mfcc(y = signal, sr = sr, n_mfcc  =n_mfcc), axis = 0)
    return mfcc_signal

def approximate_ts(X, window_size):
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    X_paa = paa.transform(X)
    return X_paa

def timeSeriesToImage(ts, size_x = None, kind = "GADF", window_size = 0):
    if window_size != 0:
        ts = approximate_ts(ts.reshape(1, -1) , window_size)
        ts = ts.reshape(-1,1)
    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')
    mtf = MarkovTransitionField()
    rp = RecurrencePlot()

    rp = RecurrencePlot()

    if kind == "GADF":
        img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "GASF":
        img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "MTF":
        img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
#         img = transformer.transform(ts)
    elif kind == "RP":
        img = rp.fit_transform(pd.DataFrame(ts).T)[0]
#         img = transformer.transform(ts)
    elif kind == "RGB_GAF":
        gasf_img = gasf.transform(pd.DataFrame(ts).T)[0]
        gadf_img = gadf.transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gasf_img,gadf_img,np.zeros(gadf_img.shape)))
    elif kind == "GASF_MTF":
        gasf_img = gasf.transform(pd.DataFrame(ts).T)[0]
        mtf_img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
        
        img = np.dstack((gasf_img,mtf_img, np.zeros(gasf_img.shape)))
    elif kind == "GADF_MTF":
        gadf_img = gadf.transform(pd.DataFrame(ts).T)[0]
        mtf_img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gadf_img,mtf_img, np.zeros(gadf_img.shape)))
    return img


# In[ ]:


def audio_mfcc(signal, sr, n_mfcc = 30):
    mfcc_signal = np.mean(librosa.feature.mfcc(y = signal, sr = sr, n_mfcc  =n_mfcc, fmin=300., fmax=600., center = True, n_mels = 20, n_fft = 1024), axis = 0)
    normalized_mfcc_feature = normalize(mfcc_signal)
    return np.array(normalized_mfcc_feature)


# def convert_dataset_to_mfcc(DATASET_FILE, n_mfcc):
#     pathlib.Path(saveTo).mkdir(parents=True, exist_ok=True)
#     pathlib.Path(os.path.join(saveTo, f.parts[-2])).mkdir(parents=True, exist_ok=True)
#     print("************")
#     print(f)
#     signal, sr = librosa.load(f, duration=6.5)
#     print("**********##########")
#     normalized_mfcc_feature = audio_mfcc(signal, sr, n_mfcc = n_mfcc)
#     # print("MFCC size: ", len(normalized_mfcc_feature), normalized_mfcc_feature)
#     np.save(f"{DATASET_FILE}{normalized_mfcc_feature}.npy")

def generate_spectrogram(aud, Fs):

        fig, ax = plt.subplots(1,1, tight_layout = True, frameon=False, figsize = (2.56,2.56))
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
        plt.axis('off')
        return fig


def extract_features(DATASET_FILE, n_mfcc,  kind, res_sig_size, features_folder):
    saveTo = f"{features_folder}/{alg}_dataset_{n_mfcc}_{res_sig_size}/"
    # pathlib.Path(saveTo).mkdir(parents=True, exist_ok=True)


    files = sorted(list(pathlib.Path(DATASET_FILE).rglob("*.wav")))
    for f in tqdm(files, total = len(files)):
        pathlib.Path(os.path.join(saveTo, f.parts[-2])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "mfcc", f.parts[-2])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "rms", f.parts[-2])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "mel_spectrogram", f.parts[-2])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "zcr", f.parts[-2])).mkdir(parents=True, exist_ok=True)

        signal, sr = librosa.load(f, duration=5.0)
        

        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512, center=True)
        spectrogram = generate_spectrogram(signal, sr)
 
        S, phase = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=512, center=True, pad_mode='constant')

        normalized_mfcc_feature = audio_mfcc(signal, sr, n_mfcc = n_mfcc)

        x = len(normalized_mfcc_feature) // res_sig_size

        img = timeSeriesToImage(normalized_mfcc_feature, size_x =  None, kind = kind, window_size = x)
        cv2.imwrite(os.path.join(saveTo, f.parts[-2], f.stem + ".png"), img)
        
        spectrogram.savefig(os.path.join(features_folder, "mel_spectrogram", f.parts[-2], f.stem + ".png"))

        np.save(os.path.join(features_folder, "mfcc", f.parts[-2], f.stem + ".npy"), rms)
        np.save(os.path.join(features_folder, "rms", f.parts[-2], f.stem + ".npy"), normalized_mfcc_feature)
        

        np.save(os.path.join(features_folder, "zcr", f.parts[-2], f.stem + ".npy"), zcr)

import common
if __name__ == "__main__":
    

    for alg in tqdm(["GADF", "GASF", "MTF", "RP", "RGB_GAF", "GASF_MTF", "GADF_MTF"]):
        extract_features(common.AUG_AUDIO_DATASET, 20, alg, 90, features_folder =  common.FEATURES_FOLDER)



# In[ ]:




