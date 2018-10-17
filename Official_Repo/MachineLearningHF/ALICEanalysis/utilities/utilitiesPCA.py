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

sys.path.insert(0, '../utilities')
from utilitiesCorrelations import *


def dimreduction(path,dataframe_sig,dataframe_bkg,varlist,n_pca,n_cases,dohistbool=True):
        print("=== Number of principal components: ",n_pca)
        print("\n--- PCA and dimentional reduction ... ")
        Sig_pca_stuff = getPCAdataframe(dataframe_sig,varlist,n_pca)
        Bkg_pca_stuff = getPCAdataframe(dataframe_bkg,varlist,n_pca)
        pca_dataframe_sig = Sig_pca_stuff[0]
        pca_dataframe_bkg = Bkg_pca_stuff[0]

        # do scatter plots on principal components
        pcaScatterFig = plt.figure(figsize=(40,40))
        pcaScatterFig.suptitle("Dim. reduction  - %d principal components"%n_pca,fontsize=75,)
        index = 1
        rowscols = getrowcol(n_cases)
        rows = rowscols[0]
        cols = rowscols[1]
        for i_col in range(1,n_pca+1):
                namex = pca_dataframe_sig.columns[i_col-1]
                for i_col2 in range(1,n_pca+1):
                        if i_col2>i_col:
                                namey = pca_dataframe_sig.columns[i_col2-1]
                                padSc = plt.subplot(rows,cols,index)
                                plt.scatter(pca_dataframe_sig[namex],pca_dataframe_sig[namey],s=3,c="red",marker="o",alpha=0.3)
                                plt.scatter(pca_dataframe_bkg[namex],pca_dataframe_bkg[namey],s=3,c="blue",marker="o",alpha=0.3)
                                padSc.set_xlabel(namex)
                                padSc.set_ylabel(namey)
                                padSc.legend(("signal","background"))
                                progressbar(index,n_cases)
                                index += 1
                                #padSc.set_title("Pearson corr. %f"%np.corrcoef(pca_dataframe[namex],pca_dataframe[namey])[0,1])
                        else:
                                continue
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        sys.stdout.flush()
        
        path_pca="%s/PCA/%d"%(path,n_pca)
        checkdir(path_pca)
        plt.savefig(path+"/%d.png"%n_pca, bbox_inches="tight")  

        # save pc DataFrames
        pca_dataframe_sig.to_pickle("%s/df_pc_sig.py"%path_pca)
        pca_dataframe_bkg.to_pickle("%s/df_pc_bkg.py"%path_pca) 
             
        if dohistbool>0:
                sys.stdout.flush()
                print("\n\t... docorrmatrix ...",end=" ")
                docorrmatrix(pca_dataframe_sig,None,"pca_signal",path_pca)
                docorrmatrix(pca_dataframe_bkg,None,"pca_background",path_pca)
                print("DONE")
                sys.stdout.flush()
                dohist(Sig_pca_stuff[1].explained_variance_ratio_,Bkg_pca_stuff[1].explained_variance_ratio_,path_pca)
        print("     DONE")
        return (pca_dataframe_sig,pca_dataframe_bkg)

def dohist(array_sig,array_bkg,path):
        sys.stdout.flush()
        print("\t... dohist ... ",end="")
        figVarRat = plt.figure(figsize=(15,15))
        
        figVarRat.suptitle("sig: %.4f, bkg: %.4f"%(dolistsum(array_sig),dolistsum(array_bkg)),fontsize=40)
        pad = plt.subplot(1,1,1)
        position_sig = [float(i) for i in frange(0.8,len(array_sig)+1-0.2,1)]
        position_bkg = [float(j) for j in frange(1.2,len(array_bkg)+1+0.2,1)]
        plt.bar(x=position_sig,height=array_sig,width=0.4,color="red")
        plt.bar(x=position_bkg,height=array_bkg,width=0.4,color="blue")
        pad.xaxis.set_ticks(range(0,len(position_sig)+1))
        pad.set_xlabel("principal component",fontsize=20)
        pad.set_ylabel("explained variance ratio",fontsize=20)
        pad.set_ylim(0,0.3)
        pad.get_xaxis().set_tick_params(labelsize=20)
        pad.get_yaxis().set_tick_params(labelsize=20)
        plt.legend(("signal","background"),fontsize=30)
        plt.savefig("%s/VarRat.png"%path)
        print("DONE")


def dolistsum(lst):
        n_Var = len(lst)
        lst_sum = 0
        for i in range(0,n_Var):
                lst_sum += lst[i]
        return lst_sum


def factorial(n):
        m = n-1
        result = n
        while m>1:
                result *= m
                m -= 1
        if n<2:
                result = 1
        return result


def frange(start, stop, step):
        i = float(start)
        lst = []
        while i < stop:
                lst.append(i)
                i += step
        return lst


def getPCAdataframe(dataframe,varlist,n_pca):
        data        = dataframe.loc[:,varlist]                          # get the dataframe with only the variables in 'varlist' 
        data_values = data.values                                       # get only the values
        data_values = StandardScaler().fit_transform(data_values)       # standardize all the values: gaussian distibution with mean=0 and sigma=1

        pca = PCA(n_pca)                                                # define the PCA object
        principalComponent = pca.fit_transform(data_values)
        
        pca_name_list = []
        for i_pca in range(1,n_pca+1):
                pca_name_list.append("princ_comp_%d"%i_pca)

        pca_dataframe = pd.DataFrame(data=principalComponent,columns=pca_name_list)
        return (pca_dataframe,pca)


def getrowcol(N):
        rowcol = []
        row = int(math.sqrt(N))
        col = N//row
        if N%row>0:
                col += 1
        rowcol.append(row)
        rowcol.append(col)
        return rowcol


def ncomb(n):
        return factorial(n)//( factorial(2)*factorial(n-2) )


def mergedf(path,df_std_sig,df_std_bkg,df_pc_sig,df_pc_bkg,Dtype,n_pca):
        print("--- Merging the DataFrames ... ",end="")
        # get original labels and give them to the new DataFrame
        index_sig = df_std_sig.index.values
        index_bkg = df_std_bkg.index.values
        df_pc_sig.index = index_sig
        df_pc_bkg.index = index_bkg

        # concatenate DataFrames to get the final ones for signal and background
        final_df_sig = pd.concat([df_pc_sig,df_std_sig],axis=1)
        final_df_bkg = pd.concat([df_pc_bkg,df_std_bkg],axis=1)

        # concatenate the last DataFrames to get the final one
        final_df = pd.concat([final_df_sig,final_df_bkg])

        # randomize row position
        final_df_shuffled = final_df.iloc[np.random.permutation(len(final_df))]

        # save
        checkdir(path)
        final_df_shuffled.to_pickle("%s/training_df_%s_pc%d.py"%(path,Dtype,n_pca))
        print("DONE")
        print("\n### Final DataFrame saved in %s ###\n"%path)
        sys.stdout.flush()

