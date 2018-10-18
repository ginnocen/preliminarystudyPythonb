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

sys.path.insert(0, '../utilities')
from utilitiesCorrelations import *
from utilitiesGeneral import *


time0 = datetime.now()

filename         = sys.argv[1]
D_species        = sys.argv[2]
pt_min           = float(sys.argv[3])
pt_max           = float(sys.argv[4])
SaveScatterPlots = sys.argv[5]

path = "./plots/%s/%.1f_%.1f_GeV"%(D_species,pt_min,pt_max)

################ training set ##################
train_set_sig, train_set_bkg = gettrainingset(filename,D_species,pt_min,pt_max,path)

######## variable distribution plots ###########
vardistplot(train_set_sig, train_set_bkg,D_species,pt_min,pt_max,path)

########### correlation matrix #################
correlationmatrix(train_set_sig,train_set_bkg,path)

############### scatter plots ##################
mylistvariables = getlistvar()          # later we shall define a function that receives the type of D-meson as argument
if SaveScatterPlots == "True":
        scatterplotsdefvar(mylistvariables,train_set_sig,train_set_bkg,"%s/Scatter_plots_def_variables"%path)

time1 = datetime.now()
howmuchtime = time1-time0
print("\n===\n===\tExecution END. Start time: %s\tEnd time: %s\t(%s)\n===\n\n\n"%(time0.strftime('%d/%m/%Y, %H:%M:%S'),time1.strftime('%d/%m/%Y, %H:%M:%S'),howmuchtime))
