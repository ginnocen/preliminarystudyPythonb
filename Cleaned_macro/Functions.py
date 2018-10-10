import array
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import sys, os
from timeit import default_timer as timer
from datetime import datetime
#from ROOT import TNtuple
#from ROOT import TH1F, TH2F, TCanvas, TFile, gStyle, gROOT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#========================================
#       Functions defined by user 
#========================================

#__________________________________________________________________________________
def CheckDir(path):
        if not os.path.exists(path):
                os.makedirs(path)
#__________________________________________________________________________________
def ProgressBar(part,tot):
        sys.stdout.flush()      # deletes what dumped in the line
        length=100
        perc = part/tot
        num_dashes = int(length*perc)
        print("\r[",end='')     # go at the beginning of the line and start writing
        # dashes (completed)
        for i in range(0,num_dashes+1):
                print("#",end='')
        # tick (missing)
        for i in range(0,length-num_dashes-1):
                print("-",end='')
        print("] {0:.0%}".format(perc),end='')
#__________________________________________________________________________________
def DoCorrMatrix(dataframe,number,string,path,c_type="pearson"):
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
        plt.savefig("%s/CorrMatrix%s_%s.pdf"%(path,c_type,string))
#__________________________________________________________________________________
def DimReduction(dataframe,varlist,n_pca,n_cases,name,dohistbool=True):

        data        = dataframe.loc[:,varlist]                          # get the dataframe with only the variables in 'varlist' 
        data_values = data.values                                       # get only the values
        data_values = StandardScaler().fit_transform(data_values)       # standardize all the values: gaussian distibution with mean=0 and sigma=1

        pca = PCA(n_pca)                                                # define the PCA object
        principalComponent = pca.fit_transform(data_values)
        
        pca_name_list = []
        for i_pca in range(1,n_pca+1):
                pca_name_list.append("Princ. comp. %d"%i_pca)

        pca_dataframe = pd.DataFrame(data=principalComponent,columns=pca_name_list)

        # do scatter plots on principal components
        pcaScatterFig = plt.figure(figsize=(15,15))
        pcaScatterFig.suptitle("Dim. reduction %s - %d principal components"%(name,n_pca),fontsize=40)
        index = 1
        rowscols = GetRowCol(n_cases)
        rows = rowscols[0]
        cols = rowscols[1]
        for i_col in range(1,n_pca+1):
                namex = pca_dataframe.columns[i_col-1]
                for i_col2 in range(1,n_pca+1):
                        if i_col2>i_col:
                                namey = pca_dataframe.columns[i_col2-1]
                                padSc = plt.subplot(rows,cols,index)
                                color=""
                                if name=="Signal":
                                        color+="red"
                                elif name=="Background":
                                        color="blue"
                                plt.scatter(pca_dataframe[namex],pca_dataframe[namey],s=3,c=color,marker="o",alpha=0.3)
                                padSc.set_xlabel(namex)
                                padSc.set_ylabel(namey)
                                index += 1
                                #padSc.set_title("Pearson corr. %f"%np.corrcoef(pca_dataframe[namex],pca_dataframe[namey])[0,1])
                        else:
                                continue
        plt.subplots_adjust(hspace=0.75,wspace=0.75)
        #plt.show()
        
        path="./%s/PCA"%name
        CheckDir(path)
        plt.savefig(path+"/%d.png"%n_pca, bbox_inches="tight")   
             
        if dohistbool>0:
                DoCorrMatrix(pca_dataframe,None,"pca_%s"%name,path)
                DoHist(pca.explained_variance_ratio_,name,path)
        #plt.show()
#__________________________________________________________________________________
def DoHist(array,name,path):
        figVarRat = plt.figure(figsize=(15,15))
        figVarRat.suptitle(name,fontsize=40)
        pad = plt.subplot(1,1,1)
        position = range(1,len(array)+1)
        col=""
        if name=="Signal":
                col+="red"
        elif name=="Background":
                col="blue"
        plt.bar(position,array,color=col)
        pad.set_xlabel("principal component",fontsize=20)
        pad.set_ylabel("explained variance ratio",fontsize=20)
        pad.set_ylim(0,0.3)
        pad.get_xaxis().set_tick_params(labelsize=20)
        pad.get_yaxis().set_tick_params(labelsize=20)
        plt.savefig("%s/VarRat.png"%path)
#__________________________________________________________________________________
def GetRowCol(N):
        rowcol = []
        row = int(math.sqrt(N))
        col = N//row
        if N%row>0:
                col += 1
        rowcol.append(row)
        rowcol.append(col)
        return rowcol
#__________________________________________________________________________________
def factorial(n):
        m = n-1
        result = n
        while m>1:
                result *= m
                m -= 1
        if n<2:
                result = 1
        return result
#__________________________________________________________________________________
def NComb(n):
        return factorial(n)//( factorial(2)*factorial(n-2) )
#__________________________________________________________________________________
def ScatterPlotsDefVar(mylistvariables,train_set_sig,train_set_bkg,path):
        CheckDir(path)
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
                        ProgressBar((i_col-1)*(num_variables_corr)+i_col2,(num_variables_corr)**2)

                        plt.savefig(path+"/scatter_%d.png"%i_col, bbox_inches="tight")














