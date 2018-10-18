from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, confusion_matrix
import seaborn as sn
import sys, os
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

from utilitiesGeneral import *


def correlationmatrix(train_set_sig,train_set_bkg,path):
        print("--- Correlation matrix of standard variables ... ",end="")
        sys.stdout.flush()
        c_type = 'pearson'      # kinds of correlation coefficients: pearson, kendall, spearman
        docorrmatrix(train_set_sig,100,"Signal",path,c_type)
        docorrmatrix(train_set_bkg,101,"Background",path,c_type)
        print("DONE")
        sys.stdout.flush()

def docorrmatrix(dataframe,number,string,path,c_type="pearson"):
        figCorr = plt.figure(num=number,figsize=(15,15))
        figCorr.suptitle("%s correlation - %s"%(c_type,string),fontsize=40)
        padcorr = plt.subplot(1,1,1)
        axis = padcorr.matshow(dataframe.corr(c_type))
        figCorr.colorbar(axis)
        padcorr.xaxis.set_ticks(range(0,len(dataframe.columns)))
        padcorr.yaxis.set_ticks(range(0,len(dataframe.columns)))
        padcorr.xaxis.set_ticks_position('bottom')
        padcorr.set_xticklabels(['']+dataframe.columns,rotation=45)
        padcorr.set_yticklabels(['']+dataframe.columns)
        final_path = "%s/%s"%(path,string)
        checkdir(final_path)
        plt.savefig("%s/CorrMatrix%s_%s.pdf"%(final_path,c_type,string))


def scatterplotsdefvar(mylistvariables,train_set_sig,train_set_bkg,path):
        print("--- Scatter plots of default variables ... ")
        checkdir(path)
        num_variables_corr = len(mylistvariables)
        for i_col in range(1,num_variables_corr+1):
                namex = mylistvariables[i_col-1]              

                index_plot_sig = 1+i_col                # signal plots 
                index_plot_bkg = 1+i_col+num_variables_corr  # background plots

                fig_scPlot     = plt.figure(num=index_plot_sig,figsize=(18,15))             
                fig_scPlot.suptitle("Scatter plot %s"%namex,fontsize=40)   

                for i_col2 in range(1,num_variables_corr+1):               
                        namey = mylistvariables[i_col2-1]

                        plt.figure(index_plot_sig)   
                        s_pad_scPlot = plt.subplot(3,3,i_col2)
                        plt.scatter(train_set_sig[namex],train_set_sig[namey],s=3,c="red",marker="o",alpha=0.3)
                        # ----- correlation coefficient -----
                        ndarray_corr_sig = np.corrcoef(train_set_sig[namex],train_set_sig[namey])
                        ndarray_corr_bkg = np.corrcoef(train_set_bkg[namex],train_set_bkg[namey])
                        s_pad_scPlot.set_title("corr. sig.: %.4f, corr. bkg.: %.4f"%(ndarray_corr_sig[0,1],ndarray_corr_bkg[0,1]))
                        # -----------------------------------
                        s_pad_scPlot.set_xlabel(namex)
                        s_pad_scPlot.set_ylabel(namey)
                        plt.subplots_adjust(wspace=0.25, hspace=0.3)

                        plt.scatter(train_set_bkg[namex],train_set_bkg[namey],s=3,c="blue",marker="o",alpha=0.3)
                        s_pad_scPlot.legend(("signal","background"))
                        progressbar((i_col-1)*(num_variables_corr)+i_col2,(num_variables_corr)**2)

                        plt.savefig(path+"/scatter_%d.png"%i_col, bbox_inches="tight")
        print("\nDONE")

def vardistplot(train_set_sig,train_set_bkg,D_species,pt_min,pt_max,path):
        print("\n--- Standard variables distributions ... ",end="")
        sys.stdout.flush()
        figure = plt.figure(figsize=(18,15))
        figure.suptitle('Variables distributions', fontsize=40)
        num_variables = len(train_set_sig.columns)
        for i_subpad in range(1,num_variables):                 # NB: the last variable is NOT PLOTTED 
                s_pad = plt.subplot(4, 3, i_subpad)                     
                name_var = train_set_sig.columns[i_subpad-1]
                plt.xlabel(name_var,fontsize=11)                
                plt.ylabel("entries",fontsize=11)               
                plt.yscale('log')
                kwargs = dict(alpha=0.3,density=True, bins=100)
                n, bins, patches = plt.hist(train_set_sig[name_var], facecolor='r', label='signal', **kwargs)          
                n, bins, patches = plt.hist(train_set_bkg[name_var], facecolor='b', label='background', **kwargs)
                s_pad.legend()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig("%s/Var_distributions.pdf"%path)
        print("DONE")
        sys.stdout.flush()
