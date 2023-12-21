# Infant crying interpretation based on deep learning

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [requirements](#requirements)
* [Launch](#launch)
* [Results](#results)

## Introduction
Crying is the natural way for infants to communicate with the world. The ability to interpret an infant’s cry and their need, especially for inexperienced parents helps understand physical, emotional, and developmental health. Its application extends to diagnoses of diseases like deafness, autism, and others. Training humans to percept infants’ cries is a difficult and long-term process, and still not that effective. Infant cry classification based on deep learning is a state-of-the-art, inexpensive, and more accurate approach than human perception. This study aims to build an accurate deep-learning model for infant cry classification. The used data is Donate Cry Corpus dataset of five classes: hungry, tired, burping, belly pain, and discomfort. Data augmentation techniques were applied to increase the data of all classes except hungry. We used Mel-spectrogram images and Time series imagining (TSI) algorithm outputs as features of our built model. Many experiments based on different image sizes for different TSI were conducted. A 30 Mel-frequency Cepstral coefficients (MFCCs) feature was generated and supplied to TSI algorithms. Using the Mel Spectrogram feature achieved 92.86% while using TSI achieved an accuracy of 71.43, 100, 49.23, 84.44, and 100% for Gramian Angular Summation Fields (GASF), Gramian Angular Difference Fields (GADF), Markov Transition Fields (MTF), Recurrence plots (RP), and RGB GAF respectively. The results show that, for different image sizes, using RGB GAF based on features achieved an accuracy of 100% on testing data. The paper is <a href="" target=
"_blank">Paper(will be added)</a>.


![Systemoveriew (1)](https://github.com/Hammoudmsh/Infant-Crying-Interpretation-based-on-Deep-Learning/assets/57059181/b539ea04-97da-4925-88cc-e005c0333663)
**Research questions:**
- 
- 
-

## Technologies
* Computer vision
* Deep learning
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
python audio_mfcc_tsi
python generateSpectrograms.py
python features_extractor.py
```

### train
```
nohup python train.py --FEATURE_NAME mfcc/  --ALGORITHMS [RF,SVC,KNN,DTC,Bagging] --DEBUG 0 --CV 10 --DATA_TYPE images --METHOD mel_spectrogram > mel_RF_SVC_KNN_DTC_Bagging.txt
```
# Results

