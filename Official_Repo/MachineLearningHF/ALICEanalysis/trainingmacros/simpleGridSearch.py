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
import uproot

import sys
sys.path.insert(0, '../utilities')
#from utilitiesRoot import FillNTuple, ReadNTuple, ReadNTupleML
from utilitiesModels import *
from utilitiesGridSearch import *

time0 = datetime.now()

neventspersample=1000
classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()

suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
train_set = pd.read_pickle("../buildsample/trainsample%s.pkl" % (suffix))
test_set = pd.read_pickle("../buildsample/testsample%s.pkl" % (suffix))
filenametest_set_ML="output/testsample%sMLdecision.pkl" % (suffix)
ntuplename="fTreeDsFlagged"

X_train= train_set[mylistvariables]
X_train_others=train_set[mylistvariablesothers]
y_train=train_set[myvariablesy]

X_test= test_set[mylistvariables]
X_test_others=test_set[mylistvariablesothers]
y_test=test_set[myvariablesy]

'''
namesCV=["Random_Forest","GradientBoostingClassifier"]
classifiersCV=[RandomForestClassifier(),GradientBoostingClassifier()]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.03,0.04,0.05,0.06,0.07], 'n_estimators': [1000,2000,3000],'max_depth' : [3,4,5]}]]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.03,0.04,0.05,0.06,0.07], 'n_estimators': [1000,2000,3000],'max_depth' : [4]}]]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.05], 'n_estimators': [1000,2000,3000],'max_depth' : [4,5,6,7,8,9,10,20,50,100]}]]
param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.05], 'n_estimators': [1000,2000,3000],'max_depth' : [2,4,6,8,10,12,14,16,18,20]}]]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.03,0.05,0.07], 'n_estimators': [500,2000,4000,8000],'max_depth' : [4]}]]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.01,0.04,0.08,0.1], 'n_estimators': [1000,2000,4000,8000],'max_depth' : [6]}]]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.01], 'n_estimators': [1000,2000,4000,8000],'max_depth' : [1,10,50]}]]
#param_gridCV = [[{'n_estimators': [50], 'max_features': [2],'max_depth': [4]}],[{'learning_rate': [0.01,0.05,0.1], 'n_estimators': [1000,2000,4000,8000],'max_depth' : [6,8,10,12,14]}]]

keys = [["param_max_features", "param_max_depth","param_n_estimators"] , ["param_learning_rate","param_max_depth","param_n_estimators"]]
changeparameter=["param_n_estimators","param_n_estimators"]
'''

namesCV=["AdaBoost","RandomForest","GradientBoostingClassifier"]
classifiersCV=[AdaBoostClassifier(),RandomForestClassifier(),GradientBoostingClassifier()]
param_gridCV = [[{'n_estimators':[100,200,300],'learning_rate':[0.1,0.5,0.9]}],[{'n_estimators': [100,200,300], 'max_features': [3],'max_depth': [4]}],[{'learning_rate': [0.05], 'n_estimators': [1000,2000,3000],'max_depth' : [4]}]]


keys = [['param_n_estimators','param_learning_rate'],["param_max_features", "param_max_depth","param_n_estimators"], ["param_learning_rate","param_max_depth","param_n_estimators"]]
changeparameter=["param_n_estimators","param_n_estimators","param_n_estimators"]

ncores=-1
#grid_search_models,grid_search_bests=do_gridsearch(namesCV,classifiersCV,mylistvariables,param_gridCV,X_train,y_train,3,ncores)
grid_search_models,grid_search_bests,dfscore=do_gridsearch(namesCV,classifiersCV,mylistvariables,param_gridCV,X_train,y_train,3,ncores)
savemodels(names,grid_search_models,"output","GridSearchCV"+suffix)

#plot_gridsearch(namesCV,changeparameter,grid_search_models,"plots",suffix)

perform_plot_gridsearch(namesCV,dfscore,keys,changeparameter,"plots",suffix)

###################### training sequence ######################
#trainedmodels=fit(names, classifiers,X_train,y_train)
#print ('Training time')
#print (datetime.now() - time0)
###################### importance study ######################
#importanceplotall(mylistvariables,names,trainedmodels,suffix)
###################### saving model ######################
#savemodels(names,trainedmodels,"output",suffix)
###################### testing sequence ######################
#time1 = datetime.now()
#test_setML=test(names,trainedmodels,X_test,test_set)
#test_set.to_pickle(filenametest_set_ML)
#print ('Testing time')
#print (datetime.now() - time1)
