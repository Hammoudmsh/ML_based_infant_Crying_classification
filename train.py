#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa


from keras.layers import *
from keras.models import *
from time import time
from keras.utils import np_utils
# import tensorflow_model_analysis as tfma

from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.grid_search import GridSearchCV,RandomizedSearchCV


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import keras.backend as K
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os, sys
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import precision_score, recall_score, roc_auc_score, make_scorer, accuracy_score,classification_report, confusion_matrix
import matplotlib.pyplot as plt

# from scikeras.wrappers import KerasClassifier
from scikeras.wrappers import KerasClassifier
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score,  accuracy_score,precision_recall_curve, recall_score, precision_score, make_scorer, precision_recall_curve
import datetime
import joblib
import pathlib
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB0


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from pactools.grid_search import GridSearchCVProgressBar

# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# In[2]:


import common
from ML_DL_utilis import MLDL_utilitis
from utilis import seeding
import parsing_file2
seeding(42)
mldl_uts = MLDL_utilitis()
# def specificity(y_true, y_pred, x):
#     """
#     param:
#     y_pred - Predicted labels
#     y_true - True labels 
#     Returns:
#     Specificity score
#     """
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     specificity = tn / (tn + fp + K.epsilon())
#     return specificity

def merge(value_1, value_2):
    return str(value_1) + "+-" + str(value_2)

def save_params_train_val_best(df, best_param, filename):
    scoring = {
    "accuracy": "accuracy", #make_scorer(accuracy_score),
    # "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    }

    columns = ["rank_test_accuracy", "params", "algorithm"]
    for met in scoring.keys():
        columns.extend(["mean_train_"+met, "std_train_"+met])

    for met in scoring.keys():
        columns.extend(["mean_test_"+met, "std_test_"+met])
    df = df[columns]
    
    
    df = df[df['params'] == best_param]

    


    df1 = pd.DataFrame()
    df1["params"] = df["params"]
    df1["algorithm"] = df["algorithm"]

    for met in scoring.keys():
        df1[met+"_train"] = df.apply(lambda row: merge(row["mean_train_"+met], row["std_train_"+met]), axis=1)
    for met in scoring.keys():
        df1[met+"_test"] = df.apply(lambda row: merge(row["mean_test_"+met], row["std_test_"+met]), axis=1)

    
    df1.to_csv(filename)
    return df1
    # with open(f'{filename}.txt', 'w') as f:
    #     f.write(str(best_param))

# In[3]:




def readParametersFromCmd(read):
    global features_name, ALGORITHMS, DEBUG, CV, DATA_TYPE, METHOD
    if read:
        parser = parsing_file2.create_parser_disease_model()
        args = parser.parse_args()

        features_name = args.FEATURE_NAME
        DEBUG = args.DEBUG
        ALGORITHMS = args.ALGORITHMS
        ALGORITHMS = ALGORITHMS.strip('][').split(",")
        DATA_TYPE = args.DATA_TYPE

        CV = args.CV

        METHOD = args.METHOD


    else:
        features_name = "mfcc"
        ALGORITHMS = ["RF"]#"LR", "Ridge", "SVC", "KNN", "XGB", "DTC", "RF", "SGD"]
        ALGORITHMS=["RF","LR","Ridge","SVC","KNN","XGB","DTC","RF","SGD","NB","MLP","Bagging"]
        DEBUG = True
        CV = 2
        DATA_TYPE = "num"
        METHOD = "GADF_dataset_20_90"

# # #------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def load_images(features_folder, subfolder):
    X = []
    y = []
    # print(list(pathlib.Path(f"{features_folder}/{subfolder}").rglob("*.png")))
    for file in list(pathlib.Path(f"{features_folder}/{subfolder}").rglob("*.png")):
        tmp2 = str(file.parent.stem)#.split("\\")[-1]
        img = image.load_img(file, target_size=(216, 216))
        x = image.img_to_array(img)
        X.append(x.flatten())
        y.append(tmp2)
    X = np.array(X)
    y = np.array(y)
    LABELS = set(y)
    print(X.shape, y.shape, LABELS)
    return X, y, LABELS

def load_np_files(features_folder):
    X = []
    y = []
    for file in list(pathlib.Path(features_folder).rglob("*.npy")):
        
        tmp1 = np.load(file)
        tmp2 = str(file.parent.stem)#.split("\\")[-1]
        X.append(tmp1.flatten())
        y.append(tmp2)
    X = np.array(X)
    y = np.array(y)
    LABELS = set(y)

    print(X.shape, y.shape, LABELS)
    return X, y, LABELS

readParametersFromCmd(True)
# features_name = "mfcc"
features_folder = f"./features/{features_name}"

current_model = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
t= "_".join(ALGORITHMS)
current_model = f"{common.ARTICLE_RESULTS}/{current_model}/{features_name}{t}/"
pathlib.Path(f'{current_model}/').mkdir(parents=True, exist_ok=True)#metrics
pathlib.Path(f'{current_model}/models').mkdir(parents=True, exist_ok=True)#metrics
pathlib.Path(f'{current_model}/train').mkdir(parents=True, exist_ok=True)#metrics

mldl_uts.setDir(current_model)



#----------------------------------------------------------------------------read data
print("-"*100, features_name, "\t", DATA_TYPE, "\t", current_model)
if DATA_TYPE == "num":
    print("case 1")
    X, y, LABELS = load_np_files(f"./features/{features_name}")
    LABELS =['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

    df = pd.DataFrame(X)
    COLUMNS = [f"S{i}" for i in df.columns]
    df.columns = COLUMNS
    df["label"] = y
    tmp = pd.get_dummies(df['label'])
    df1 = pd.concat([df, tmp], axis = 1)

    #----------------------------------------------------------------------------data split
    seeding(42)
    trainValIdx, testIdx = train_test_split(list(range(len(df1))), test_size = 0.2, random_state = 42, shuffle=True, stratify = df1['label'])
    print("(Train + val)=Train data size: ", len(trainValIdx))
    print("Test data size: ", len(testIdx))
    df_train = df1.loc[trainValIdx]
    df_test = df1.loc[testIdx]
    # df = df1.sample(frac = 1)
    # df = df1.sample(frac = 1)
    LABELS =['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

    tmp = LABELS.copy()
    tmp.append('label') # ['label', 'belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    X_train = df_train.drop(tmp, axis = 1)
    y_train = df_train['label']

    X_test = df_test.drop(tmp, axis = 1)
    y_test = df_test['label']

    y_train_enc = df_train[LABELS]
    y_test_enc = df_test[LABELS]
elif DATA_TYPE == 'images':
    print("case 2")
    X, y, LABELS = load_images(common.FEATURES_FOLDER, METHOD)
    seeding(42)
    trainValIdx, testIdx = train_test_split(list(range(len(X))), test_size = 0.2, random_state = 42, shuffle=True, stratify = y)
    print("train + val data size: ", len(trainValIdx))
    print("test data size: ", len(testIdx))
    LABELS =['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

    X_train = X[trainValIdx]
    X_test = X[testIdx]

    y_train = y[trainValIdx]
    y_test = y[testIdx]

    y_train_enc = pd.get_dummies(y_train).values




#----------------------------------------------------------------------------EDA





# # Machine learning 

# In[12]:



# def f1_categorical(y, y_pred, **kwargs):
#     return f1_score(y.argmax(1), y_pred, **kwargs)

# def precision_categorical(y, y_pred, **kwargs):  
#     return precision_score(y.argmax(1), y_pred.argmax(1), **kwargs)

# def recall_categorical(y, y_pred, **kwargs):
#     return recall_score(y.argmax(1), y_pred.argmax(1), **kwargs)

# def accuracy_categorical(y, y_pred, **kwargs):
#     return accuracy_score(y.argmax(1), y_pred, **kwargs)
CV_record_all = pd.DataFrame()



Results = {"train":{}, "test":{}}
scoring = {
    "accuracy": "accuracy", #make_scorer(accuracy_score),
    "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision_micro": make_scorer(precision_score, labels=range(len(LABELS)), average="micro"),
    "precision_macro": make_scorer(precision_score, labels=range(len(LABELS)), average="macro"),
    "precision_weighted": make_scorer(precision_score, labels=range(len(LABELS)), average="weighted"),

    "recall_micro": make_scorer(recall_score, labels=range(len(LABELS)), average="micro"),
    "recall_macro": make_scorer(recall_score, labels=range(len(LABELS)), average="macro"),
    "recall_weighted": make_scorer(recall_score, labels=range(len(LABELS)), average="weighted"),

#     "roc_auc_score": make_scorer(roc_auc_score)
}


# ## ------------------------------------------------------------------------------------------Logistic Regression

# In[14]:

# try:
if 1:
    method = "LR"
    if method in ALGORITHMS:
        print("-----------")
        model = LogisticRegression(multi_class='multinomial', class_weight = "balanced", random_state = 42)

        param_grid  = [
            # {
            #     'solver' : ['liblinear'],
            #     'penalty' : ['l1', 'l2'],
            #     'max_iter' : [50,100,200,500],
            #     'C' : [0.001, 0.01, 0.1, 1]
            # }, #no support for multinomial
            # {
            #     'solver' : ['saga'],
            #     'penalty' : ['elasticnet', 'l1', 'l2', 'none'],
            #     'max_iter' : [50,100,200,500],
            #     'C' : [0.001, 0.01, 0.1, 1],
            #     'l1_ratio' : [1]
            # },
            {
                'LR__solver' : ['saga'],
                'LR__penalty' : ['l1'],
                'LR__max_iter' : [10],
                'LR__C' : [0.2, 1],
                'LR__l1_ratio' : [1]
            },
            {
            'LR__solver' : ['lbfgs', 'sag', 'newton-cg', 'lbfgs'],
            'LR__penalty' : ['l2', 'none'],
            'LR__max_iter' : [50,100,200,500],
            'LR__C' : [0.001, 0.01, 0.1, 1]
            },

        # add more parameter sets as needed...
        ]
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("LR", LogisticRegression(multi_class='multinomial', class_weight = "balanced", random_state = 42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method

        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=True).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")


        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')

        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)

        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

# except Exception as e:
#     print("ERROR:          ", e)

# ## ------------------------------------------------------------------------------------------Ridge Classifier


try:
    method = "Ridge"
    if method in ALGORITHMS:
        model = RidgeClassifier(class_weight = "balanced", random_state = 42)

        param_grid  = [
            {
            'Ridge__solver' : ["lbfgs"],
            'Ridge__alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'Ridge__positive' : [True]
            },
            {
            'Ridge__solver' : ["saga", "svd", "cholesky", "lsqr", "sparse_cg", "sag" ],
            'Ridge__alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # 'l1_ratio' : [1]
            },

        # add more parameter sets as needed...
        ]

        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("Ridge", RidgeClassifier(class_weight = "balanced", random_state = 42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=True).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')

        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")
except Exception as e:
    print("ERROR:          ", e)
# ## ------------------------------------------------------------------------------------------DecisionTreeClassifier


try:
    method = "DTC"
    if method in ALGORITHMS:

        model = DecisionTreeClassifier(criterion="entropy", class_weight = "balanced", random_state=42)


        param_grid = dict(
            DTC__max_depth = range(1,21),
            DTC__max_leaf_nodes = [20, 30, 40, 50],
            DTC__max_features = [6, 12, 18, 24]
                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("DTC", DecisionTreeClassifier(criterion="entropy", class_weight = "balanced", random_state=42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')
        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)

# ## ------------------------------------------------------------------------------------------RandomForestClassifier


# try:
if 1:
    method = "RF"
    if method in ALGORITHMS:

        model = RandomForestClassifier(criterion="entropy", class_weight = "balanced", random_state=42)

        # solver  = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']#'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']   

        param_grid = dict(
            RF__max_depth = range(1,21),
            RF__max_leaf_nodes = [50],
            RF__n_estimators = [10, 100, 200],
            RF__max_features = ['sqrt', 'log2'],
                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("RF", RandomForestClassifier(criterion="entropy", class_weight = "balanced", random_state=42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')

        Results["test"][method] = pd.DataFrame.from_dict(x).T
        #######display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

# except Exception as e:
#     print("ERROR:          ", e)

# ## ------------------------------------------------------------------------------------------GaussianNB


try:
    method = "NB"
    if method in ALGORITHMS:
        model = GaussianNB()

        # solver  = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']#'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']   

        param_grid = dict(
            NB__var_smoothing = np.logspace(0,-9, num=100)
                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("NB", GaussianNB())])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')

        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)

# ## ------------------------------------------------------------------------------------------MLPClassifier not


try:
    method = "MLP"
    if method in ALGORITHMS:

        model = MLPClassifier(max_iter=500, batch_size = 32, activation='relu', shuffle = False, random_state = 42)


        param_grid = dict(
            MLP__hidden_layer_sizes = [(50,50,50), (50,100,50), (100,)],
            MLP__activation = ['identity', 'logistic', 'tanh', 'relu'],
            MLP__solver = ['sgd', 'adam', 'lbfgs'],
            MLP__alpha = [0.0001, 0.05],
            MLP__learning_rate = ['constant','adaptive', 'invscaling'],
                    )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("MLP", MLPClassifier(max_iter=500, batch_size = 32, activation='relu', shuffle = False, random_state = 42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')


        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)

# ## ------------------------------------------------------------------------------------------KNeighborsClassifier

try:
    method = "KNN"
    if method in ALGORITHMS:

        model =  KNeighborsClassifier()

        param_grid = dict(
            KNN__n_neighbors = range(1,21, 2),
            KNN__weights = ['uniform', 'distance'],
            KNN__metric = ['euclidean'],#, 'manhattan', 'minkowski']
            # pca__n_components = [15, 20],
                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("KNN", KNeighborsClassifier())])
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = y_test,#np.argmax(y_test, axis = 1),
            y_pred = y_pred,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(y_test,y_pred, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')
        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)
# ## ------------------------------------------------------------------------------------------Stochastic Gradient Boosting or Gradient Boosting Machine (GBM) 
try:
    method = "SGD"
    if method in ALGORITHMS:

        model =  GradientBoostingClassifier(random_state=42)

        param_grid = dict(
            SGD__n_estimators = [5, 25, 50, 75, 100],
            SGD__learning_rate = [0.001, 0.01, 0.1, 1],
            SGD__subsample = [0.5, 0.7, 1.0],
            SGD__max_depth = [7, 9, 11],
                         )
        param_grid = {} if DEBUG else param_grid

        #-----------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("SGD", GradientBoostingClassifier(random_state=42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        aa = y_test#(np.argmax(y_test_enc.values, axis = 1))
        bb = y_pred#(np.argmax(y_pred, axis = 1))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = aa,
            y_pred = bb,
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(aa,bb, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')


        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)
# ## ------------------------------------------------------------------------------------------XGBClassifier

try:
    method = "XGB"
    if method in ALGORITHMS:

        model =  XGBClassifier()

        #     reg_alpha = hp.quniform('reg_alpha', 40,180,1),
        #     reg_lambda = np.random.uniform(0,1,10),
        #     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        #     'min_child_weight' : np.random.uniform(0,10,3),
        seeding(42)
        param_grid = dict(
            XGB__max_depth = range(1, 11, 2),
            XGB__gamma = np.random.uniform(0,1,3),
            XGB__n_estimators = range(40, 50),
            XGB__tree_method = ["hist"],
            XGB__learning_rate = [0.1, 0.001, 0.001],
            XGB__seed = [42]
                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("XGB", XGBClassifier())])
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        print(y_train_enc)
        grid_result = grid.fit(X_train, y_train_enc.values)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        # aa = (np.argmax(y_test_enc.values, axis = 1))
        aa = pd.from_dummies(y_test_enc).values
        bb = (np.argmax(y_pred, axis = 1))
        bb = [LABELS[e] for e in bb]

        image_cm = mldl_uts.make_confusion_matrix(
            y = aa,
            y_pred = bb,
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(aa,bb, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')



        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)
# ## ------------------------------------------------------------------------------------------SVC
try:
    method = "SVC"
    if method in ALGORITHMS:

        model =  SVC(class_weight = 'balanced', random_state = 42)

        param_grid = dict(
            SVC__kernel = ['linear'],#', 'poly', 'rbf', 'sigmoid'],#precomputed
            SVC__degree=[5, 7, 8, 9],
            SVC__C= [0.001, 0.1,1, 10, 100],
            SVC__gamma = [1,0.1,0.01,0.001],
                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("SVC", SVC(class_weight = 'balanced', random_state = 42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        aa = y_test#(np.argmax(y_test.values, axis = 0))
        bb = y_pred#(np.argmax(y_pred, axis = 0))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = aa,
            y_pred = bb,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(aa,bb, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')




        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)


# ## ------------------------------------------------------------------------------------------Bagged Decision Trees
try:
    method = "Bagging"
    if method in ALGORITHMS:

        model =  BaggingClassifier(random_state=42)

        param_grid = dict(
            Bagging__n_estimators = [25, 50, 75, 100, 200],

                         )
        param_grid = {} if DEBUG else param_grid

        #-------------------------------------------------------------------
        scaler = StandardScaler() 
        model = Pipeline(steps=[("scaler", StandardScaler()),   ("Bagging", BaggingClassifier(random_state=42))])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = CV,
                           scoring = scoring,
                            refit='accuracy',
                            return_train_score=True,
                            error_score="raise",
                            verbose=2
                           )
        grid_result = grid.fit(X_train, y_train)
        grid_result.cv_results_["algorithm"] = method
        df = pd.DataFrame(grid_result.cv_results_)
        df = df.sort_values(by=['mean_test_accuracy'], ascending=False).reset_index().round(4)

        df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
        # ####display(df)
        Results["train"][method] = df
        # df.to_csv("results.csv")

        model = grid_result.best_estimator_ 
        #------------------------------------------confusion matrix
        y_pred = model.predict(X_test.astype('float32'))

        aa = y_test#(np.argmax(y_test.values, axis = 0))
        bb = y_pred#(np.argmax(y_pred, axis = 0))

        # y_pred = y_pred.argmax(axis=1)
        image_cm = mldl_uts.make_confusion_matrix(
            y = aa,
            y_pred = bb,
        #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
            cmap = "gray",#"Greys",
            categories = LABELS,
            figsize = (15,10),
            title = "Confusion matrix",
                show = False, prefix = method +"_", save = False, return_cm = True
        );
        x = classification_report(aa,bb, output_dict = True, target_names = LABELS)
        joblib.dump(grid_result.best_estimator_ , f'{current_model}/models/gs_model_{method}.pkl')
        x["model_size"] = os.path.getsize(f'{current_model}/models/gs_model_{method}.pkl')



        Results["test"][method] = pd.DataFrame.from_dict(x).T
        ####display(Results["test"][method])
        print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
        CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
        CV_record_all = pd.concat([CV_record_all, CV_record])
        Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

except Exception as e:
    print("ERROR:          ", e)



# ## ------------------------------------------------------------------------------------------ANN

# In[26]:


# def f1(y_true, y_pred):


# def get_hp():
#     batches = [10, 20, 40, 60, 80, 100]
#     batches = [16, 32]
#     # epochs = [10, 50, 100]
#     # epochs = [25, 50, 100]#100, 200]
#     epochs = [30, 50, 100]
#     optimizers = ['SGD', 'Adam', 'RMSprop']#, 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
# #     optimizers = ['Adam']
#     learning_rate = [0.01]
# #     learning_rate = [0.01]
#     # momentum1 = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#     # init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#     init = ['uniform']

#     # activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#     # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     # neurons = [25, 30]
#     param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, learning_rate = learning_rate)
#     param_grid = dict(model__optimizer=optimizers, 
#                       optimizer__learning_rate=learning_rate,
#                      model__epochs = epochs,
#                      model__loss = ['categorical_crossentropy']
#                      )
#     return param_grid


# def create_model(epochs, optimizer, loss):
#     model = Sequential()
#     model.add(Dense(512, input_shape=(216,)))
#     model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
#     model.add(Dropout(0.1))   # Dropout helps protect the model from memorizing or "overfitting" the training data
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(5))
#     model.add(Activation('softmax')) # This special "softmax" a
#     # if is_compile is False:
#     #     return model
#     model.compile(loss=loss)
#     # , optimizer=optimizer, #binary_crossentropy
#     #               metrics=[
#     #                   'accuracy',
# #                            tf.keras.metrics.TruePositives(),
# #                            tf.keras.metrics.TrueNegatives(),
# #                            tf.keras.metrics.FalsePositives(),
# #                            tf.keras.metrics.FalseNegatives(),
# #                            tfa.metrics.F1Score(name = "f1_weighted", num_classes=10, average='weighted',threshold=0.5),
# #                            tfa.metrics.F1Score(name = "f1_micro", num_classes=10, average='micro', threshold=0.5),
# #                            tfa.metrics.F1Score(name = "f1_macro", num_classes=10, average='macro', threshold=0.5),
# #                            specificity,
# #                            tf.keras.metrics.Precision(),
# #                            tf.keras.metrics.Recall(),
# #                            tf.keras.metrics.AUC(curve = 'ROC'),
# #                            tf.keras.metrics.AUC(curve = 'PR')
#                  #          ],
#                  # ) 
#     return model


# def create_model1(epochs, optimizer):#, learning_rate = 0.001, neurons=50, is_compile = True, epochs = 1):
#     classes_num =5

#     model = Sequential()
#     model.add(Dense(256, input_shape=(216,)))
#     model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
#     model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
# #     model.add(Dense(128))
# #     model.add(Activation('relu'))
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(classes_num))
#     model.add(Activation('softmax')) # This special "softmax" a
#     model.compile(loss='categorical_crossentropy')
# #     if is_compile is False:
# #         return model
# #     model.compile(loss='categorical_crossentropy', optimizer=optimizer, #binary_crossentropy
# #                   metrics=[
# #                       'accuracy',
# # #                            tf.keras.metrics.TruePositives(),
# # #                            tf.keras.metrics.TrueNegatives(),
# # #                            tf.keras.metrics.FalsePositives(),
# # #                            tf.keras.metrics.FalseNegatives(),
# # #                            tfa.metrics.F1Score(name = "f1_weighted", num_classes=10, average='weighted',threshold=0.5),
# # #                            tfa.metrics.F1Score(name = "f1_micro", num_classes=10, average='micro', threshold=0.5),
# # #                            tfa.metrics.F1Score(name = "f1_macro", num_classes=10, average='macro', threshold=0.5),
# # #                            specificity,
# # #                            tf.keras.metrics.Precision(),
# # #                            tf.keras.metrics.Recall(),
# # #                            tf.keras.metrics.AUC(curve = 'ROC'),
# # #                            tf.keras.metrics.AUC(curve = 'PR')
# #                           ],
# #                  ) 
#     return model

# # In[27]:

# try:
#     method = "ANN"
#     if method in ALGORITHMS:

#         seeding(42)
#         seed = 42 # can be any number, and the exact value does not matter
#         np.random.seed(seed)

#         nb_classes = len(LABELS)
#         current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
#         log_dir = "logs/hparam_tuning/" + current_time
#         start= time()

#         model = KerasClassifier(model=create_model,verbose=2, random_state = 42, shuffle=False)

#         param_grid  = get_hp()
#         print(param_grid)
#         #-------------------------------------------------------------------
#         # model = create_model()



#         grid = GridSearchCV(estimator=model, param_grid=param_grid , cv = CV,
#                            scoring = scoring,
#                             refit='accuracy',
#                             return_train_score=True,
#                             error_score="raise", 
#                            )
#         grid_result = grid.fit(X_train.values, y_train_enc.values, shuffle=False, callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)])
#         grid_result.cv_results_["algorithm"] = method
#         df = pd.DataFrame(grid_result.cv_results_)
#         df = df.sort_values(by=['mean_test_accuracy'], ascending=True).reset_index().round(4)

#         df = df.drop(['index', "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"], axis=1)
#         # ####display(df)
#         Results["train"][method] = df
#         # df.to_csv("results.csv")

#         model = grid_result.best_estimator_ 
#         #------------------------------------------confusion matrix


#         y_pred = model.predict(X_test.astype('float32'))


#         aa = pd.from_dummies(y_test_enc).values

#         bb = (np.argmax(y_pred, axis = 1))
#         bb = [LABELS[e] for e in bb]
#         # print(aa.shape)
#         aa = aa.flatten()
#         bb = np.array(bb).flatten()

#         # print(aa, bb)
#         # print(aa.shape, bb.shape)

#         # print(bb.shape[0])
#         # for i in range(bb.shape[0]):
#         #     aa[i] = LABELS.index(aa[i])
#         #     bb[i] = LABELS.index(bb[i])



#         image_cm = mldl_uts.make_confusion_matrix(
#             y = aa,#np.argmax(y_test, axis = 1),
#             y_pred = bb,
#         #         group_names = ['True Neg','False Pos','False Neg','True Pos'],
#             cmap = "gray",#"Greys",
#             categories = LABELS,
#             figsize = (15,10),
#             title = "Confusion matrix",
#                 show = False, prefix = method +"_", save = False, return_cm = True
#         );
#         x = classification_report(aa,bb, output_dict = True, target_names = LABELS)
#         joblib.dump(grid_result.best_estimator_ , f'{common.ARTICLE_RESULTS}/gs_model_{method}.pkl')
#         x["model_size"] = os.path.getsize(f'{common.ARTICLE_RESULTS}/gs_model_{method}.pkl')

#         Results["test"][method] = pd.DataFrame.from_dict(x).T
#         ####display(Results["test"][method])
#         print("Val-Accuracy: ", grid_result.best_score_, grid_result.best_params_,)
#         CV_record = save_params_train_val_best(Results["train"][method], grid_result.best_params_, f'{current_model}/{method}.csv')
#         CV_record_all = pd.concat([CV_record_all, CV_record])
        # Results["test"][method].to_csv(f"{current_model}/test_results_{method}.csv")

# except Exception as e:
#     print("ERROR:          ", e)

# # In[28]:















#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------





wanted = ['params', 'mean_test_accuracy', 'std_test_accuracy','mean_train_accuracy', 'std_train_accuracy',
       'algorithm']
train_results_df = pd.DataFrame()
for experiment in Results["train"].keys():
    train_table = Results["train"][experiment]
#     train_table = test_table.rename(columns= {col_name: experiment + "_" + col_name for col_name in train_table.columns})
    train_table.to_csv(f"{current_model}/train/train_results_{experiment}.csv")
    tmp = train_table[wanted]
    train_results_df = pd.concat([train_results_df, tmp], axis = 0)
train_results_df.to_csv(f"{current_model}/train_val_results_all.csv")

CV_record_all.to_csv(f"{current_model}/CV_record_all.csv")


test_results_df = pd.DataFrame()
for experiment in Results["test"].keys():
    test_table = Results["test"][experiment]
    test_table = test_table.rename(columns= {col_name: experiment + "_" + col_name for col_name in test_table.columns})
    test_results_df = pd.concat([test_results_df, test_table], axis = 1)
# display(test_results_df)
test_results_df.to_csv(f"{current_model}/test_results.csv")


# In[29]:


res = {}
for experiment in Results["test"].keys():
    acc, _, _ = test_results_df.loc["accuracy"][[experiment+"_precision", experiment+"_recall", experiment+"_f1-score"]]
    macro_prc, macro_recall, macro_f1 = test_results_df.loc["macro avg"][[experiment+"_precision", experiment+"_recall", experiment+"_f1-score"]]
    weighted_prc, weighted_recall, weighted_f1 = test_results_df.loc["weighted avg"][[experiment+"_precision", experiment+"_recall", experiment+"_f1-score"]]
    
    
#     res[experiment] = [acc, macro_prc, macro_recall, macro_f1, weighted_prc, weighted_recall, weighted_f1]
    res[experiment] = {
                        "acc":acc,
                        "macro_prc":macro_prc,
                        "macro_recall":macro_recall,
                        "macro_f1":macro_f1,
                        "micro_prc":macro_prc,
                        "micro_recall":macro_recall,
                        "micro_f1":macro_f1,
                        "weighted_prc":weighted_prc,
                        "weighted_recall":weighted_recall,
                        "weighted_f1":weighted_f1
    }
    
    #     res[experiment]["acc"] = acc
#     res[experiment]["macro_prc"] = macro_prc
#     res[experiment]["macro_recall"] = macro_recall
#     res[experiment]["macro_f1"] = macro_f1
#     res[experiment]["weighted_prc"] = weighted_prc
#     res[experiment]["weighted_re?call"] = weighted_recall
#     res[experiment]["weighted_f1"] = weighted_f1
    
res = pd.DataFrame.from_dict(res, orient ='index')#, columns = ["acc", "macro_prc", "macro_recall", "macro_f1", "weighted_prc", "weighted_recall", "weighted_f1"])
# fig = plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)
res.plot(kind='bar', title="Testing results", ax = ax)
plt.legend(bbox_to_anchor=(1, 1))
plt.grid()
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1, 0.1)
minor_ticks = np.arange(0, 1, 0.01)


# plt.xticks(major_ticks)
# plt.xticks(minor_ticks, minor=True)
plt.yticks(major_ticks)
plt.yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.savefig(f"{current_model}/test_results_graph.png")

