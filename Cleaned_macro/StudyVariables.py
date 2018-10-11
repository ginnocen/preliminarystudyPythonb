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

import Functions as func                # user defined functions
import D_mesons_variables as Dvars      # D mesons variables




time0 = datetime.now()
filename         = sys.argv[1]
SaveScatterPlots = sys.argv[2]
n_pca_variables  = int(sys.argv[3])

# check if there are the directories where to save the files, otherwise create them
path_Sig = "./Signal"
path_Bkg = "./Background"
func.CheckDir(path_Sig)
func.CheckDir(path_Bkg)

# ===== variables to be checked for correlations =====
#mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"] 
mylistvariables = Dvars.D_dictionary["Ds"]   
#train_set = pd.read_pickle("../testsample.pkl")                    # load objects stored in the file specified in the path ---> it is a DataFrame
train_set = pd.read_pickle(filename)                            # load objects stored in the file specified in the path ---> it is a DataFrame
print("\n=== Opened file: ",filename)
train_set_sig=train_set.loc[train_set['signal_ML'] == 1]        # .loc function used to return a DataFrame with all the info about the entries satisfying the condition in brackets     
train_set_bkg=train_set.loc[train_set['signal_ML'] == 0]

#==================================
#   Variable distribution plots
#==================================
print("\n--- Standard variables distributions ... ",end="")
sys.stdout.flush()
figure = plt.figure(figsize=(18,15))
figure.suptitle('Variables distributions', fontsize=40)
num_variables = len(train_set_sig.columns)
for i_subpad in range(1,num_variables):                 # NB: the last variable is NOT PLOTTED 
        s_pad = plt.subplot(4, 3, i_subpad)             # it means: the figure is subdivided in 4x3 subplots and I am selecting the number i_subpad (the i_subpad th one! The numerations strarts from 1)        
        name_var = train_set_sig.columns[i_subpad-1]
        plt.xlabel(name_var,fontsize=11)                # equivalent to s_pad.set_xlabel
        plt.ylabel("entries",fontsize=11)               # equivalent to s_pad.set_ylabel
        plt.yscale('log')
        kwargs = dict(alpha=0.3,density=True, bins=100)
        n, bins, patches = plt.hist(train_set_sig[name_var], facecolor='r', label='signal', **kwargs)           # NB: this function returns MORE THAN ONE ARGUMENT
        n, bins, patches = plt.hist(train_set_bkg[name_var], facecolor='b', label='background', **kwargs)
        s_pad.legend()
plt.subplots_adjust(hspace=0.5)
plt.savefig("Var_distributions.pdf")
print("DONE")
sys.stdout.flush()

#========================
#   Correlation matrix
#========================
print("--- Correlation matrix of standard variables ... ",end="")
sys.stdout.flush()
c_type = 'pearson'      # kinds of correlation coefficients: pearson, kendall, spearman
func.DoCorrMatrix(train_set_sig,100,"signal",path_Sig,c_type)
func.DoCorrMatrix(train_set_bkg,101,"background",path_Bkg,c_type)
print("DONE")
sys.stdout.flush()

#============================================
#   Scatter plots among standard variables
#============================================
if SaveScatterPlots == "True":
        print("--- Scatter plots of default variables ... ")
        func.ScatterPlotsDefVar(mylistvariables,train_set_sig,train_set_bkg,"./Scatter_plots_def_variables")
        print("\nDONE")

#=====================================================
#   Dimentional reduction with principal components
#=====================================================
print("--- Principal components extraction ... ",end="")
sys.stdout.flush()

if n_pca_variables>len(mylistvariables):
        n_pca_variables = len(mylistvariables)

func.DimReduction(train_set_sig,train_set_bkg,mylistvariables,n_pca_variables,func.NComb(len(mylistvariables)))
print("DONE")
sys.stdout.flush()

time1 = datetime.now()
howmuchtime = time1-time0
print("\n===\n===\tExecution END. Start time: %s\tEnd time: %s\t(%s)\n==="%(time0.strftime('%d/%m/%Y, %H:%M:%S'),time1.strftime('%d/%m/%Y, %H:%M:%S'),howmuchtime))





