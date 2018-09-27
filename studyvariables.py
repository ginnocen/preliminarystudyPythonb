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
from ROOT import TNtuple
from ROOT import TH1F, TH2F, TCanvas, TFile, gStyle, gROOT


time0 = datetime.now()
            
mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]
train_set = pd.read_pickle("trainsample.pkl")

train_set_sig=train_set.loc[train_set['signal_ML'] == 1]
train_set_bkg=train_set.loc[train_set['signal_ML'] == 0]


X_test= train_set_sig[mylistvariables]


figure = plt.figure(figsize=(15,15))
ax = plt.subplot(4, 3, 1)  
plt.xlabel('d_len_xy_ML',fontsize=11)
plt.ylabel("entries",fontsize=11)
plt.yscale('log')
kwargs = dict(alpha=0.3,density=True, bins=100)
n, bins, patches = plt.hist(train_set_sig['d_len_xy_ML'], facecolor='b', label='signal', **kwargs)
n, bins, patches = plt.hist(train_set_bkg['d_len_xy_ML'], facecolor='g', label='background', **kwargs)
ax.legend()
plt.show()
