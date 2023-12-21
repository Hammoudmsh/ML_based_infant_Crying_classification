import csv
import pandas as pd
# import dataframe_image as dfi
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from IPython.display import display



import os
import time
import random
import numpy as np
import cv2
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
import pandas as pd
from collections import defaultdict

import seaborn as sn
from pathlib import Path
from PIL import Image




class utilitis:
    def compareTwoList(self, a, b):
        matches = [idx for idx, item in enumerate(zip(a, b)) if item[0] == item[1]]
        matchesNum = len(matches)
        return matches, matchesNum

    def save2csv(self, fileName, data, cols, header = False):
        with open(fileName, 'w+', newline='', encoding='utf-8') as f:
            write = csv.writer(f, delimiter=',')
            if header:
                write.writerow(cols)
            write.writerows(data)
    def find_between(self, s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""

    def find_between_r(self, s, first, last ):
        try:
            start = s.rindex( first ) + len( first )
            end = s.rindex( last, start )
            return s[start:end]
        except ValueError:
            return ""

    def isContain(self, fileName, fileTypes):
            for ext in fileTypes:
                if  ext not in fileName:
                    return False
            return True

    def show(self, df, nr):
        with pd.option_context('display.max_rows', nr,
                           'display.max_columns', None,
                           'display.width', 800,
                           'display.precision', 3,
                           'display.colheader_justify', 'left'):
            display(df)
        
    def tensor(self, *x):
        tmp = []
        for i in x:
            tmp.append(np.array(i))
    #         tmp.append(tf.convert_to_tensor(i))
        return tmp

    def dataframeAsImage(self, d, path, rowNames, save, colsNames =None):
        df = pd.DataFrame(data=d, index = rowNames, columns = colsNames)
        if save:
            dfi.export(df, path)
        return df
    def showRow(self, display_list, title, size = None):
        # plt.figure()
        fig, ax = plt.subplots(1, len(display_list), figsize = size)
        for i in range(len(display_list)):    
            if display_list[i] is not None:
                ax[i].set_title(title[i])
                ax[i].imshow(tf.keras.utils.array_to_img(display_list[i]));
                ax[i].axis('off')
                plt.close()
        # plt.show()
        return fig
        df = pd.DataFrame(data=d, index = rowNames)
        if save:
            dfi.export(df, path)

    def display(self, display_list, idx = None, num = None, title =  None, size =(10, 10), show = True):    
        if len(display_list[0].shape) in [2,3] :
            f = self.showRow(display_list, title, size = size)
            return f
        else:
            if idx is  None and num is not None or num == 1:
                idx = np.random.randint(0, len(display_list[0]), num)
            fig, ax = plt.subplots(num, len(display_list), figsize = size)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            for j, i in enumerate(idx):
                if j ==0:
                    titles__ = title
                else:
                    titles__ = [""] * len(display_list)
                tmp = []
                for img in display_list:
                    if img is not None and i < len(img):
                        x = img[i]
                    else:
                        x = None
                    tmp.append(x)
                    
                for i in range(len(display_list)):    
                    if tmp[i] is not None:
                        ax[j][i].set_title(titles__[i])
                        if i  in [1,2]:
                            ax[j][i].imshow(tf.keras.utils.array_to_img(tmp[i]), cmap = 'jet');
                        else:
                            ax[j][i].imshow(tf.keras.utils.array_to_img(tmp[i]));
                        ax[j][i].axis('off')
                        ax[j][i].set_aspect('equal')

                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                        if show == False:
                            plt.close()
            return fig



def plot_img(img, title="", cmap= "gray", figsize = (5,5)):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.title(title)

def to3D(img, SIZE = None):
    # if SIZE is not None:
    # img =  cv2.resize(img, (64,64))

    if img.ndim == 2:
        # img = np.expand_dims(img, axis=2)
        img = np.dstack((img, img, img))
    return img



def concat_images(*images):
    """Generate composite of all supplied images."""
    # Get the widest width.
    width = max(image.width for image in images)
    # Add up all the heights.
    padding = 10
    height = sum(image.height+padding for image in images) - padding
    composite = Image.new('RGB', (width, height))
    # Paste each image below the one before it.
    y = 0
    for image in images:
        composite.paste(image, (0, y))
        y += image.height + padding
    return composite

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result



def display_image_grid(A, B = None, C = None, num = 5, color = (255, 255, 255), size = (256, 256), direction = [0,1]):
    m = min(num, len(A))
    wanted = np.random.randint(0,len(A) , m)
    
    col1 = [add_margin(Image.fromarray(cv2.resize(A[i], size)), 10, 10, 10, 10, color = color) for i in wanted]

    if B is not None:
        col2 = [add_margin(Image.fromarray(cv2.resize(B[i], size)), 10, 10, 10, 10, color = color) for i in wanted]
    col3 = [add_margin(Image.fromarray(cv2.resize(C[i], size)), 10, 10, 10, 10, color = color) for i in wanted]
    
    col1 = np.concatenate(col1, axis = direction[0])
    col3 = np.concatenate(col3, axis = direction[0])
    if B is not None:
        col2 = np.concatenate(col2, axis = direction[0])
        res = np.concatenate([col1, col2, col3], axis=direction[1])
    else:
        res = np.concatenate([col1, col3], axis=direction[1])
    return res

""" Seeding the randomness. """
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

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



class epochs_logger:
    def __init__(self, metrics_names, float_precision = 3):
        self.metrics_names = metrics_names
        self.history = pd.DataFrame(columns = self.metrics_names)
        self.float_precision = float_precision
        self.reset()
        
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
        self.history = pd.DataFrame(columns = self.metrics_names)
    
    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] = val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["val"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()])
    
    def update_history(self, values):
        self.history.loc[len(self.history.index)] = values
    

    def update_history_in(self):
        # print(self.metrics_names)
        to_add = []
        for metr in self.metrics_names:
            if metr in self.metrics.keys():
                val = self.metrics[metr]["val"]
            else:
                val = None
            to_add.append(val)
        self.update_history(to_add)

    def get_history(self):
        return self.history
# epochs_log = epochs_logger(metrics_names = ['loss', 'iou', 'dice_score', 'f1_score', 'val_loss', 'val_iou', 'val_dice', 'val_f1_score']) 

# # epochs_log.update_history([1,2,3,4,5, 6, 7, 448])
# # epochs_log.update_history([1,2,3,4,5, 6, 7, 8])
# epochs_log.update("loss", 22)
# epochs_log.update("iou", 221)
# #
# epochs_log.update_history_in()

# epochs_log.update("iou", 99)
# epochs_log.update_history_in()
# epochs_log.update("val_iou", 11)
# epochs_log.update_history_in()

# print(epochs_log)
# epochs_log.get_history()
# # print(epochs_log)



# concat_images(img, img)
# img = add_margin(Image.fromarray(img), 10, 10, 10, 10, color = (255, 255, 255))





def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image