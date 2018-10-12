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
        sys.stdout.flush()      
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
def GetPCADataFrame(dataframe,varlist,n_pca):
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
#__________________________________________________________________________________
def DimReduction(dataframe_sig,dataframe_bkg,Dtype,pt_min,pt_max,varlist,n_pca,n_cases,dohistbool=True):

        Sig_pca_stuff = GetPCADataFrame(dataframe_sig,varlist,n_pca)
        Bkg_pca_stuff = GetPCADataFrame(dataframe_bkg,varlist,n_pca)
        pca_dataframe_sig = Sig_pca_stuff[0]
        pca_dataframe_bkg = Bkg_pca_stuff[0]

        #print("\nPCA data frames")
        #print("%s\nlen = %d\n"%(pca_dataframe_sig,len(pca_dataframe_sig)))
        #print("%s\nlen = %d\n"%(pca_dataframe_bkg,len(pca_dataframe_bkg)))

        # do scatter plots on principal components
        pcaScatterFig = plt.figure(figsize=(40,40))
        pcaScatterFig.suptitle("Dim. reduction  - %d principal components"%n_pca,fontsize=75,)
        index = 1
        rowscols = GetRowCol(n_cases)
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
                                index += 1
                                #padSc.set_title("Pearson corr. %f"%np.corrcoef(pca_dataframe[namex],pca_dataframe[namey])[0,1])
                        else:
                                continue
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        
        path="./%s/%.1f_%.1f_GeV/PCA/%d"%(Dtype,pt_min,pt_max,n_pca)
        CheckDir(path)
        plt.savefig(path+"/%d.png"%n_pca, bbox_inches="tight")  

        # save pc DataFrames
        pca_dataframe_sig.to_pickle("%s/df_pc_sig.py"%path)
        pca_dataframe_bkg.to_pickle("%s/df_pc_bkg.py"%path) 
             
        if dohistbool>0:
                DoCorrMatrix(pca_dataframe_sig,None,"pca_signal",path)
                DoCorrMatrix(pca_dataframe_bkg,None,"pca_background",path)
                DoHist(Sig_pca_stuff[1].explained_variance_ratio_,Bkg_pca_stuff[1].explained_variance_ratio_,path)

        return (pca_dataframe_sig,pca_dataframe_bkg)
#__________________________________________________________________________________
def DoHist(array_sig,array_bkg,path):
        figVarRat = plt.figure(figsize=(15,15))
        
        figVarRat.suptitle("sig: %.4f, bkg: %.4f"%(DoListSum(array_sig),DoListSum(array_bkg)),fontsize=40)
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
#__________________________________________________________________________________
def frange(start, stop, step):
        i = float(start)
        lst = []
        while i < stop:
                lst.append(i)
                i += step
        return lst
#__________________________________________________________________________________
def DoListSum(lst):
        n_Var = len(lst)
        lst_sum = 0
        for i in range(0,n_Var):
                lst_sum += lst[i]
        return lst_sum
#__________________________________________________________________________________
def MergeDF(df_std_sig,df_std_bkg,df_pc_sig,df_pc_bkg,Dtype,pt_min,pt_max,n_pca):
        # get original labels and give them to the new DataFrame
        index_sig = df_std_sig.index.values
        index_bkg = df_std_bkg.index.values
        df_pc_sig.index = index_sig
        df_pc_bkg.index = index_bkg
        #print(df_pc_sig,"\n")
        #print(df_pc_bkg)
        # concatenate DataFrames to get the final ones for signal and background
        final_df_sig = pd.concat([df_pc_sig,df_std_sig],axis=1)
        final_df_bkg = pd.concat([df_pc_bkg,df_std_bkg],axis=1)
        #print("\n\t*** Final DataFrames for signal and background ***\n")
        #print(final_df_sig.columns)
        #print(final_df_bkg.columns)
        #print(final_df_sig)
        #print(final_df_bkg,"\n")
        # concatenate the last DataFrames to get the final one
        final_df = pd.concat([final_df_sig,final_df_bkg])
        #print("\n\t*** Final DataFrame ***\n")
        #print(final_df)
        # randomize row position
        final_df_shuffled = final_df.iloc[np.random.permutation(len(final_df))]
        #print("\n\t*** Final DataFrame shuffled***\n")
        #print(final_df_shuffled)
        # save
        path="./%s/%.1f_%.1f_GeV/PCA/%d"%(Dtype,pt_min,pt_max,n_pca)
        CheckDir(path)
        final_df_shuffled.to_pickle("%s/training_df_%s_pc%d.py"%(path,Dtype,n_pca))






