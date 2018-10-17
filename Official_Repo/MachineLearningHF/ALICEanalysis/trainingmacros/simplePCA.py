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
from utilitiesPCA import *

time0 = datetime.now()

filename         = sys.argv[1]
D_species        = sys.argv[2]
pt_min           = float(sys.argv[3])
pt_max           = float(sys.argv[4])
n_pca_variables  = int(sys.argv[5])

path = "./plots/%s/%.1f_%.1f_GeV"%(D_species,pt_min,pt_max)
path_out = "./output"

##################### training set #####################
train_set_sig, train_set_bkg = gettrainingset(filename,D_species,pt_min,pt_max,path)

mylistvariables = getlistvar()          # later we shall define a function that receives the type of D-meson as argument

############# PCA and dimentional reduction ############
pca_df = dimreduction(path,train_set_sig,train_set_bkg,mylistvariables,n_pca_variables,ncomb(len(mylistvariables)))

##### merging standard variables and pc DataFrames #####
mergedf(path_out,train_set_sig,train_set_bkg,pca_df[0],pca_df[1],D_species,n_pca_variables)

time1 = datetime.now()
howmuchtime = time1-time0
print("\n===\n===\tExecution END. Start time: %s\tEnd time: %s\t(%s)\n==="%(time0.strftime('%d/%m/%Y, %H:%M:%S'),time1.strftime('%d/%m/%Y, %H:%M:%S'),howmuchtime))

