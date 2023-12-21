
import os
import time
import random
import numpy as np
import cv2
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sn
from pathlib import Path



plt.rcParams["axes.labelsize"] = 'medium'
plt.rcParams["axes.titlecolor"] = 'red'
plt.rcParams["axes.titlesize"] = 'large'
#plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 18

""" Seeding the randomness. """

def append2csv(filename, df):
    df.to_csv(filename, mode='a', index=False, header = not Path(filename).exists())


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def confusionMatrix(y_true, y_pred):
    classes = ["a", "b", "c", "d", "e"]
    fig, ax = plt.subplots(nrows = 1, figsize = (12,7))

#         tensor.detach().numpy()
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index = [c for c in classes], columns = [c for c in classes])
#         ax.set_title(title = "confusion_matrix")
    ax.set_xlabel("ground_truth")
    ax.set_ylabel("prediction")
    sn.heatmap(df_cm, annot = True)
    return fig


def plotGraphs(axisLabel, title, labels, xyLim = [[0,2]], *listt):
        fig,ax = plt.subplots(nrows = 1, figsize=(12,5))
        plt.rcParams["figure.figsize"] = (10, 6)

        ax.set_xlabel(axisLabel[0])
        ax.set_ylabel(axisLabel[1])
        ax.set(title = title)
        for i, graphData in enumerate(listt):
            ax.plot(graphData, label = labels[i])
            plt.ylim(xyLim[1])

        # plt.xticks(np.arange(len(listt[0])))
        plt.xticks(np.arange(0, len(listt[0]), 5))

        plt.legend()
        # plt.grid()
        plt.show()
        fig.savefig("Results\\"+title +'.png')
        plt.close()
        return ax

def plotHistory(history, n = [1, 2], size = (5,5), show = False, prefix = "", titleSize = 14, labelSize = 14,  where2save = "Results/"):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    what = []
    for k in history.keys():
        if 'val' not in k:
            what.append(k)
    for metric in what:
        try:
            plt.style.use("ggplot")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)   
            if what =="loss":
                plt.semilogy(history.epoch, history[metric], color=colors[n[0]], label='Train Loss' + label)
                plt.semilogy(history.epoch, history['val_'+metric], color=colors[n[1]], label='Val Loss' + label, linestyle="--")
            else:
                plt.plot(history.epoch, history[metric], color=colors[n[0]], label='Train '+metric)
                plt.plot(history.epoch, history['val_'+metric], color=colors[n[1]], label='Val '+metric, linestyle="--")
            if what =="learning rate":
                plt.plot(history.epoch, metric, color = colors[n[0]], label= metric)
            plt.title(metric, fontsize = titleSize)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend(loc='upper left');
            plt.xlabel('Epoch', fontsize = labelSize)
            plt.ylabel(f'{metric}', fontsize = labelSize)
            labels = ax1.get_xticklabels() + ax1.get_yticklabels()
            [label.set_fontsize(15) for label in labels]
            [label.set_fontweight('bold') for label in labels]
            plt.savefig(f"{where2save}{prefix}{metric}.png")
        except:
            continue
        if show == False:
            plt.close()
