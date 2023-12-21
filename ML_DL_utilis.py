# import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import cv2

# plt.rcParams["axes.labelsize"] = 'medium'
# plt.rcParams["axes.titlecolor"] = 'black'
# plt.rcParams["axes.titlesize"] = 'large'
# #plt.rcParams["figure.figsize"] = (15, 10)
# plt.rcParams["font.size"] = 14
# plt.rcParams['axes.titlepad'] = 18





class MLDL_utilitis:
    def __init__(self, where2save = None):
        if where2save is None:
            self.where2save = ""
        else:
            self.where2save = where2save
        self.setTitle(ts = 22, tc = "black", ls = 20)


    def setDir(self, d):
        self.where2save = d

    def setTitle(self, ts = 18, tc = "black", ls = 15):
        self.titleSize = ts
        self.titleColor = tc
        self.labelSize = ls

    def searchForOne(self, listt, dictt):
        for e in listt:
            if e in dictt:
                return True
        return False

    def plotHistory(self, history, n = [1, 2], size = (5,5), show = False, prefix = "", return_plots = False):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        what = []
        for k in history.keys():
            if 'val' not in k:
                what.append(k)
        if 'epochs' not in history.keys():
            x = history[list(history.keys())[0]]
            history['epochs'] = list(range(len(x)))


        plots = {}
        for metric in what:
            plt.style.use("ggplot")
            fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)   
            if metric =="loss":
                plt.semilogy(history["epochs"], history[metric], color=colors[n[0]], label='Train Loss')
                if 'val_'+metric in history.keys():
                    plt.semilogy(history["epochs"], history['val_'+metric], color=colors[n[1]], label='Val Loss', linestyle="--")
            else:
                plt.plot(history["epochs"], history[metric], color=colors[n[0]], label='Train '+metric)
                if 'val_'+metric in history.keys():
                    plt.plot(history["epochs"], history['val_'+metric], color=colors[n[1]], label='Val '+metric, linestyle="--")
            
            if metric =="learning rate":
                plt.plot(history["epochs"], metric, color=colors[n[0]], label= metric)
            plt.title(metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend(loc='upper left');
            x = self.where2save
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel(f'{metric}', fontsize=15)
            labels = ax1.get_xticklabels() + ax1.get_yticklabels()
            [label.set_fontsize(13) for label in labels]
            [label.set_fontweight('bold') for label in labels]
            plt.savefig(f"{x}{prefix}{metric}.png")
            if return_plots:
                plots[metric] = cv2.imread(f"{x}{prefix}{metric}.png")
            if show == False:
                plt.close()
            
        if return_plots:
            return plots
    # def plotHistory(self, history, size = (12,5), show = True):

    #     plt.style.use("ggplot")
    #     txt = {"lr":"Learning Rate",
    #     "accuracy":"Train Acc",
    #     "loss":"Train Loss",
    #     "val_accuracy":"Val Acc",
    #     "val_loss":"Val loss"
    #     }

    #     for p in history.keys():
    #         if "acc" not in p and "loss" not in p:
    #             # fig = plt.subplots(1, 1, figsize = size, sharey=True)
    #             fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)

    #             plt.plot(history[p], label = p)
    #             plt.xlabel('epoch')
    #             plt.ylabel(p)
    #             if p in txt:
    #                 s = txt[p]
    #             else:
    #                 s = p
                    
    #             plt.title(s, fontsize=self.titleSize, color=self.titleColor)
    #             plt.savefig(f"{self.where2save}{p}.png")
    #             if show == False:
    #                 plt.close()

    #     if 'accuracy' in history or 'val_accuracy' in history:
    #         # fig = plt.subplots(1, 1, figsize = size, sharey=True)
    #         fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)

    #         if 'accuracy' in history:
    #             s = txt['accuracy']
    #             plt.plot(history['accuracy'], label = s, color = 'g')
    #         if 'val_accuracy' in history:
    #             s = txt['val_accuracy']
    #             plt.plot(history['val_accuracy'], label =s)
    #         plt.title('model accuracy', fontsize=self.titleSize, color=self.titleColor)
    #         plt.ylabel('accuracy')
    #         plt.xlabel('epoch')
    #         plt.ylim((0, 100))
    #         if 'accuracy' in history and 'val_accuracy' in history:
    #             plt.legend(loc='upper left');
    #         plt.savefig(self.where2save + "accuracy.png")
    #         if show == False:
    #             plt.close()

    #     if 'loss' in history or 'val_loss' in history:
    #         # fig = plt.subplots(1, 1, figsize = size, sharey=True)
    #         fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)


    #         if 'loss' in history:
    #             s = txt['loss']
    #             plt.plot(history['loss'], label = s, color = 'r')
    #         if 'val_loss' in history:
    #             s = txt['val_loss']
    #             plt.plot(history['val_loss'], label =s,color = 'b')
    #         plt.title('model loss', fontsize=self.titleSize, color=self.titleColor)
    #         plt.ylabel('loss')
    #         plt.xlabel('epoch')
    # #         plt.ylim((0, 1))
    #         if 'loss' in history and 'val_loss' in history:
    #             plt.legend(loc='upper left');
    #         plt.savefig(self.where2save + "loss.png")
    #         if show == False:
    #             plt.close()

    # def plotHistory(self, history, size = (12,5), show = True):
    #     plt.style.use("ggplot")
    #     txt = {"lr":"Learning Rate"}

    #     mine = True
    #     checkAccTrain = ["Training Accuracy","accuracy"]
    #     checkAccVal = ["Validation Accuracy", "val_accuracy"]

    #     checkLossTrain = ["Training Loss", "val_loss"]
    #     checkLossVal = ["Validation Loss", "val_loss"]


    #     for p in history.keys():
    #         if p  not in checkAccTrain and p  not in checkAccVal and p  not in checkLossTrain and p  not in checkLossVal:
    #             # fig = plt.subplots(1, 1, figsize = size, sharey=True)
    #             fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)

    #             plt.plot(history[p], label = p)
    #             plt.xlabel('epoch')
    #             plt.ylabel(p)
    #             if p in txt:
    #                 s = txt[p]
    #             else:
    #                 s = p
                    
    #             plt.title(s, fontsize=self.titleSize, color=self.titleColor)
    #             plt.savefig(f"{self.where2save}{p}.png")
    #             if show == False:
    #                 plt.close()

    #     if self.searchForOne(checkAcc, history):
    #         # fig = plt.subplots(1, 1, figsize = size, sharey=True)
    #         fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)
    #         if mine == True:
    #             accTrainList = history['accuracy']
    #             accValList = history['val_accuracy']
    #         else:
    #             pass



    #         if self.searchForOne(checkAcc[0:2], history):
    #             plt.plot(accList, label = "accuracy", color = 'g')
    #         if self.searchForOne(checkAcc[2:4], history):
    #             plt.plot(accValList, label ="val_accuracy")
    #         plt.title('model accuracy', fontsize=self.titleSize, color=self.titleColor)
    #         plt.ylabel('accuracy')
    #         plt.xlabel('epoch')
    #         plt.ylim((0, 1))
            
    #         # if 'accuracy' in history and 'val_accuracy' in history:
    #         plt.legend(loc='upper left');

    #         plt.savefig(self.where2save + "accuracy.png")
    #         if show == False:
    #             plt.close()
        
    #     if self.searchForOne(checkLoss, history):
    #         # fig = plt.subplots(1, 1, figsize = size, sharey=True)
    #         fig, ax1 = plt.subplots(1, 1, figsize=size, tight_layout=True)
    #        if mine == True:
    #             lossTrainList = history['loss']
    #             lossValList = history['val_loss']

    #         if elf.searchForOne(checkAcc[0:2], history):
    #             plt.plot(history['loss'], label = "loss", color = 'r')
    #         if elf.searchForOne(checkAcc[0:2], history):
    #             plt.plot(history['val_loss'], label ="val_loss",color = 'b')
    #         plt.title('model loss', fontsize=self.titleSize, color=self.titleColor)
    #         plt.ylabel('loss')
    #         plt.xlabel('epoch')
    # #         plt.ylim((0, 1))
    #         # if 'loss' in history and 'val_loss' in history:
    #         plt.legend(loc='upper left');
    #         plt.savefig(self.where2save + "loss.png")
    #         if show == False:
    #             plt.close()
#     def Plot_cm(self, y_predictions, y_test, class_names, show = True):
#         conf_mat = confusion_matrix(y_test, y_predictions)

#         fig = plt.subplots(1, 1, figsize = (5,5), sharey=True)
#         tick_marks = np.arange(len(class_names))
# #         plt.xticks(tick_marks,class_names)
# #         plt.yticks(tick_marks,class_names)

#         sns.heatmap(pd.DataFrame(conf_mat),annot=True,cmap="Blues", fmt='.2%', cbar=False)#fmt="d"
# #         ax.xaxis.set_label_position('top')
#     #     plt.tight_layout()
#         plt.ylabel('ground_truth')
#         plt.xlabel('Prediction');
#         plt.savefig(self.where2save + "cm.png")
#         if show == False:
#             plt.close()

#     def calcSenseSpecAcc(self, y_predictions, y_test):
#         bin_predictions = y_predictions.flatten()
#         report= classification_report(y_test, bin_predictions)
#         return report

#         conf_mat = confusion_matrix(y_test, y_predicted)
#         total = sum(sum(conf_mat))
#         sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
#         specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
#         accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total
#         return {"sensitivity":sensitivity,
#                "specificity":specificity,
#                "accuracy":accuracy
#                }
    def save(self, model,  fileName = "feedforwardnet.pth"):
        torch.save(model.state_dict(), fileName)
        print("Trained feed forward net saved at " + fileName)

    def plotGraphs(self, axisLabel, title, labels, xyLim = [[0,2]], *listt):
        fig,ax = plt.subplots(nrows = 1, figsize=(12,5))
        plt.rcParams["figure.figsize"] = (10, 6)

        ax.set_xlabel(axisLabel[0])
        ax.set_ylabel(axisLabel[1])
        ax.set(title = title, fontsize=self.titleSize, color=self.titleColor)
        for i, graphData in enumerate(listt):
            ax.plot(graphData, label = labels[i])
            plt.ylim(xyLim[1])

        plt.xticks(np.arange(len(listt[0])))
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig("Results\\"+title +'.png')
        return ax

    def plotCategories(self, X, Y, xlbl = "", ylbl = "", title = "", ax = None):
        if ax is  None:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_axes([0,0,1,1])
        New_Colors = ['green','blue','purple','brown','teal','red','black']
        ax.bar(X, Y, color=New_Colors,  width=0.3)
        ax.set_xlabel(xlbl, fontsize=20)
        ax.set_ylabel(ylbl, fontsize=20)
        ax.set_title(title, fontsize=self.titleSize, color=self.titleColor)
#         ax.set_xticks(rotation=45)
        #plt.xticks(rotation=90)

        # plt.grid(True)
        ax.set_ylim([0,max(Y)+50])

        rects = ax.patches

        # Make some labels.

        labels = [f"{Y[i]}" for i in range(len(rects))]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
            )
            ax.set_ylim(ymin = 0, ymax = max(Y)+50)


        if ax is  None:
            plt.show()
            return ax
    """
    def dataSetPlot(self, className = None, classNum = None, ax = None):
        if ax is  None:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_axes([0,0,1,1])
        New_Colors = ['green','blue','purple','brown','teal','red','black']
        ax.bar(className, classNum, color=New_Colors, width=0.3)
        ax.set_xlabel('Classes', fontsize=20)
        ax.set_ylabel('Number of Records', fontsize=20)
        ax.set_title(f"Dataset Classes ({sum(classNum)}) records")
        #plt.xticks(rotation=90)
        # plt.grid(True)
        ax.set_ylim([0,max(classNum)+50])
        rects = ax.patches

        # Make some labels.
        labels = [f"{classNum[i]}" for i in range(len(rects))]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
            )
            #plt.ylim([0, max(classNum)+70])
            ax.set_ylim(ymin = 0, ymax = max(classNum)+50 )

        if ax is  None:
            plt.show()
        return ax
    """
    def plotDetails(self,  ax = None):
        d ={}
        t = []
        cc = 0
        for p, subdirs, files in os.walk(self.directory2save):
            for s in subdirs:
                d[s] = len((glob(p + "//" + s +"//*.avi")))
                t.append(s.split("_")[0])


        self.events = set(t)

        xxx = []
        y = []
        for classN in  self.classes:
            x = []
            x.append(classN)
            eN = []
            for e in self.events:
                eN.extend(self.getContainFromDict(d, e, classN))

            x.extend(eN)
            xxx.append(x)
            y.extend(eN)

        columns = ["classes"]
        columns.extend(self.events)
        print(y)
        x = np.array(y)
        y = x.reshape((3,6)).T.flatten()

        labels = [f"{i}" for i in y]


        # create data
        df = pd.DataFrame(xxx,columns = columns)

        # plot grouped bar chart
        df.plot(x='classes',
                kind='bar',
                stacked = False,
                title=f"Dataset Classes ({sum(y)}) records",
                width = 0.7,
                ax = ax)
        rects = ax.patches
        # Make some labels.

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
            )
        ax.set_ylim(ymin = 0, ymax = max(y)+50 )
        #ax.ylim([0, max(y)+50])
        return ax

    def make_confusion_matrix(self,
                              y,
                              y_pred,
                              group_names=None,
                              categories='auto',
                              count=True,
                              show = True ,
                              percent=True,cbar=True, xyticks=True, xyplotlabels=True, sum_stats=True, figsize=None, cmap='Blues',title=None, prefix = "", save = True, return_cm = True):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''

        cf = confusion_matrix(y, y_pred)

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))

            #if it is a binary confusion matrix, show some more stats
            if len(cf)==2:
                #Metrics for Binary Confusion Matrices
                precision = cf[1,1] / sum(cf[:,1])
                recall    = cf[1,1] / sum(cf[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}    Precision={:0.3f}\nRecall={:0.3f}    F1 Score={:0.3f}".format(
                    accuracy,precision,recall,f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""


        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False


        # MAKE THE HEATMAP VISUALIZATION
        # plt.figure(figsize=figsize)
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, linewidths=3, linecolor='white', annot_kws={"size": 16})
        ax1.set_xticklabels(ax1.get_xmajorticklabels(), fontsize = 14)
        ax1.set_yticklabels(ax1.get_ymajorticklabels(), fontsize = 14)

        if xyplotlabels:
            plt.ylabel('Ground truth')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title, fontsize=self.titleSize, color=self.titleColor)
        if save:
            plt.savefig(f"{self.where2save}{prefix}cm.png")
        if return_cm:
            plt.savefig(f"{self.where2save}{prefix}cm.png")
            plt.savefig(f"cm.png")

        if show == False:
            plt.close()
        if return_cm:
            return cv2.imread(f"{self.where2save}{prefix}cm.png")
    """
    def saveModelArchitecture(self, model, fn, save = True):
        # 1&1&(1,1,1)\\ \hline
        where2save = self.where2save
        # where2save = "."
        
        model1_layers_names=[layer.name for layer in model.layers]
        model1_layers_types=[layer.__class__.__name__ for layer in model.layers]
        model1_layers_shapes=[layer.output_shape for layer in model.layers]
        model1_layers_shapes = [tuple(xi for xi in x if xi is not None) for x in model1_layers_shapes]
        s = "" 
        for element in z:
    #         print(element, list(element))
            t = list(element)
            t[-1] = f"{t[-1]}"
            s = s + "&".join(t) + "\\\ \\hline "    
        df = pd.DataFrame(data = list(zip(model1_layers_names, model1_layers_types, model1_layers_shapes)),
                          columns=["Name", "type", "Output Size"])
    #     print(model1_layers_names)
        if save:
            pass
            #dfi.export(df, f"{where2save}/{fn}") 
        
        text_file = open(f"{where2save}/{fn}.txt", "w")
        text_file.write(s) 
        text_file.close()
        
        return df
        """


    def save_model(self, fileName, epochs, model, optimizer, criterion):
        """
        Function to save the trained model to disk.
        """
        print(f"Saving final model...")
        torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, fileName)

# mldl_uts.plotCategories(X = ["a","b","c"], Y = [10, 12, 30], ax = None, xlbl = "X", ylbl = "Y", title = "TITLE");
    
    def saveModelArchitecture(self, model, fn, save = True):
        # 1&1&(1,1,1)\\ \hline
        where2save = self.where2save
        # where2save = "."
        model1_layers_names=[layer.name for layer in model.layers]
        model1_layers_types=[layer.__class__.__name__ for layer in model.layers]
        model1_layers_shapes=[layer.output_shape for layer in model.layers]
        model1_layers_shapes = [tuple(xi for xi in x if xi is not None) for x in model1_layers_shapes]
        s = "" 
        z = zip(model1_layers_names, model1_layers_types, model1_layers_shapes)
        for i, element in enumerate(z):
            t = list(element)
            t[-1] = f"{t[-1]}"
            if i == 0:
                t[-1] = t[-1][2:-3]            
            else:
                t[-1] = t[-1][1:-1]
            
            if "None" in t[-1]:
                t[-1]=  t[-1].replace("None,", "" )

            s = s + " & ".join(t) + "\\\ \\hline " +"\n"   
        df = pd.DataFrame(data = list(zip(model1_layers_names, model1_layers_types, model1_layers_shapes)),
                          columns=["Name", "type", "Output Size"])
        if save:
            
            #dfi.export(df, f"{where2save}/{fn}")
            pass

        text_file = open(f"{where2save}/{fn}.txt", "w")
        text_file.write(s) 
        text_file.close()
        return df
        