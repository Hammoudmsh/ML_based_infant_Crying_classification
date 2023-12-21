import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
# from pp_utils import *
from utils import *
import pathlib
import math
from utilis import *
import common

def generate(where):

    wanted_size = (600, 600)
    border_white = (255, 255, 255)
    border_black = (255,255,255)
    
    A = []
    B = []
    C = []

    ALGS = [
        "mfcc_GADF_dataset_20_90",
        "mfcc_GASF_dataset_20_90",
        "mfcc_GADF_dataset_20_90",
        "mfcc_MTF_dataset_20_90",
        "mfcc_GADF_MTF_dataset_20_90",
        "mfcc_GASF_MTF_dataset_20_90",
        "mfcc_RGB_GAF_dataset_20_90",
        "mfcc_RP_dataset_20_90",
        "mel_spectrogram"       

    ]
    for alg in ALGS:
        A = []
        for lbl in ["belly_pain", "burping", "discomfort", "hungry", "tired"]:
            seeding(42)
            w = np.random.randint(0,10, 1)

            xx = sorted(list(pathlib.Path(f"{common.FEATURES_FOLDER}/{alg}/{lbl}").rglob("*.png")))
            xx = xx[w[0]]
            a = cv2.imread(str(xx)); a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = add_margin(Image.fromarray(255*to3D(a)), 5, 5, 5, 5, color = border_white) 
            A.append(255*np.asarray(a))
#         res1 = display_image_grid(A, B = None, C = None, num = 4)
        res1 = np.concatenate(A, axis=1)
        res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
        parts = alg.split("_")
        if "dataset" in parts:
            z = "_".join(parts[0:parts.index("dataset")])
        else:
            z = alg
        if alg != "mel_spectrogram":
            cv2.imwrite(f"{common.ARTICLE_RESULTS}/tsi_{z}.png", 255*res1)
        else:
            cv2.imwrite(f"{common.ARTICLE_RESULTS}/tsi_{z}.png", res1)



if __name__ == "__main__":
    pathlib.Path(f'{common.ARTICLE_RESULTS}').mkdir(parents=True, exist_ok=True)#metrics
    generate(where = f'{common.ARTICLE_RESULTS}')


