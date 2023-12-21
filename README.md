# Machine Learning-based Infant Crying Interpretation

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [requirements](#requirements)
* [Launch](#launch)
* [Results](#results)

## Introduction

Crying is an inevitable character throughout the growth of infants, with conditions such as parents around them being understandable or the opposite being the case. This cry can be treated as an audio signal that carries a message about the infant’s feelings. The primary infant caregiver requires traditional ways of understanding these feelings. These feelings can be discomfort, hunger, sickness, etc. Failing to understand them correctly can cause a severe problem. Many solutions are attempting to solve this problem using different methods; however, due to the lack of proper audio feature representation and  classifier used, the achieved result requires further improvement. In this research, time-, frequency-, and time frequency-domain feature representations are used to gain in-depth information about the audio signal. Time-domain features include Zero-crossing rate (ZCR), Root Mean Square (RMS), frequency-domain features include Mel-spectrogram and time frequency-domain features include Mel-frequency Cepstral coefficients (MFCCs). Moreover, Time series imagining (TSI) algorithms are applied to transform 20-Mel-frequency Cepstral coefficients (MFCCs) features into images using different algorithms: Gramian Angular Difference Fields (GADF), Gramian Angular Summation Fields (GASF), Markov Transition Fields (MTF), Recurrence plots (RP), and RGB GAF. Then, these features are provided to different ML classifiers, such as Decision tree (DT), Random forest (RF), K nearest neighbors (KNN), and bagging. Using MFCCs, ZCR, and RMS as features achieved high performance, outperforming SOTA. Optimal parameters are found via the grid search method. Using 10-fold cross-validation, Our approach MFCCs-based RF classifier achieved an accuracy of 96.39%, outperforming the State-of-the-art Architecture of the scalogram-based shuffleNet classifier with an accuracy of 95.17%. The paper is <a href="" target="_blank">Paper(will be added)</a>.

**Research questions:**
- 
- 
-

## Technologies
* Machine learning
* time-series analysis

## Requirements
1- create a virtual environment with conda and activate it
```
conda create --name sk_pro python=3.6
conda activate sk_pro
pip install -r requirements_paper.txt
```
## Launch
### download data

* Manual downloading:
<a href="https://github.com/gveres/donateacry-corpus" target="_blank">donateacry-corpus</a>
* Automatic run scripts, to download opened, navgaze, natural, and NN human mouse datasets.
```
python download_data.py
```
### Data augmentation 
```
python data_aug.py
```
### Transform audio into images and generate Mel-spectrogram
```
python features_extractor.py
```

### train
```
<<<<<<< HEAD
python train.py --FEATURE_NAME mfcc/  --ALGORITHMS [RF] --DEBUG 0 --CV 10 --DATA_TYPE images --METHOD mfcc_mel_spectrogram
=======
nohup python train.py --FEATURE_NAME mfcc/  --ALGORITHMS [RF,SVC,KNN,DTC,Bagging] --DEBUG 0 --CV 10 --DATA_TYPE images --METHOD mel_spectrogram > mel_RF_SVC_KNN_DTC_Bagging.txt
>>>>>>> 5005698 (First commit)
```
# Results

