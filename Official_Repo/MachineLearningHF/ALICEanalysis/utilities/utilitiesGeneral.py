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


def checkdir(path):
        if not os.path.exists(path):
                os.makedirs(path)


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


def getlistvar():
        mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"] 
        #mylistvariables = Dvars.D_dictionary[D_species]
        return mylistvariables


def gettrainingset(filename,D_species,pt_min,pt_max,path):
        # check if there are the directories where to save the files, otherwise create them
        path_Sig = "%s/Signal"%path
        path_Bkg = "%s/Background"%path
        checkdir(path_Sig)
        checkdir(path_Bkg)
        ############ variables to be checked for correlations ############   
        train_set = pd.read_pickle(filename)                            
        print("\n=== Opened file: ",filename)
        print("=== D meson considered: ",D_species)
        print("=== pT interval considered: %.1f < pT < %.1f GeV/c"%(pt_min,pt_max))
        train_set_sig=train_set.loc[(train_set['signal_ML'] == 1) & (train_set['pt_cand_ML']>pt_min) & (train_set['pt_cand_ML']<pt_max)]   
        train_set_bkg=train_set.loc[(train_set['signal_ML'] == 0) & (train_set['pt_cand_ML']>pt_min) & (train_set['pt_cand_ML']<pt_max)]
        return train_set_sig, train_set_bkg


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


def progressbar(part,tot):
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

