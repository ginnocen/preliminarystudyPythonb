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
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn_evaluation import plot
import itertools


def do_gridsearch(namesCV_,classifiersCV_,mylistvariables_,param_gridCV_,X_train_,y_train_,cv_,ncores):
  grid_search_models_=[]
  grid_search_bests_=[]
  list_scores=[]    # new part for info storing
  for nameCV, clfCV, gridCV in zip(namesCV_, classifiersCV_,param_gridCV_):
    grid_search = GridSearchCV(clfCV, gridCV, cv=cv_,scoring='neg_mean_squared_error',n_jobs=ncores)
    grid_search_model=grid_search.fit(X_train_, y_train_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): # cvres["params"] is a list of dictionaries! Every dictionary contains the value for each parameter the model has run with
      print(np.sqrt(-mean_score), params)
    list_scores.append(pd.DataFrame(cvres)) # new part for info storing
    grid_search_best=grid_search.best_estimator_.fit(X_train_, y_train_)
    grid_search_models_.append(grid_search_model)
    grid_search_bests_.append(grid_search_best)

  #return grid_search_models_,grid_search_bests_
  return grid_search_models_,grid_search_bests_,list_scores

def plot_gridsearch(namesCV_,change_,grid_search_models_,output_,suffix_):

  for nameCV,change,gridCV in zip(namesCV_,change_,grid_search_models_):
    figure = plt.figure(figsize=(10,10))
    plot.grid_search(gridCV.grid_scores_, change=change,kind='bar')      # GMI original version (ok till sklearn 0.17)
    plt.title('Grid search results '+ nameCV, fontsize=17)
    plt.ylim(-0.3,0)
    plt.ylabel('negative mean squared error',fontsize=17)
    plt.xlabel(change,fontsize=17)
    plotname=output_+"/GridSearchResults"+nameCV+suffix_+".png"
    plt.savefig(plotname)


#
# --- necessary alternative to plot_gridsearch ---
#

def makestring(tpl):
  lst = list(tpl)
  #print(lst)
  string = ""
  for i in range(0,len(lst)):
    string+=str(lst[i])
    if(i<len(lst)-1):
      string+="/"
  return string

def make_lst_str_set(set1,set2):
  print("--- make_lst_str")
  comb = list(itertools.product(set1,set2))
  # list of strings
  lst_str = []
  for tpl in comb:
    lst_str.append(makestring(tpl))
  print(lst_str)
  print(type(lst_str))
  return lst_str

def make_lst_str_2(sets,ix,lst_str):
  print("--- make_lst_str_2 ",ix)
  i=ix
  lst_str_2=[]
  print(type(lst_str))
  print(type(sets[i]))
  comb=list(itertools.product(lst_str,sets[i]))
  for tpl in comb:
    lst_str_2.append(makestring(tpl))
  print("--- after appending")
  print(lst_str_2)
  i+=1
  if(i==len(sets)):
    print("\t---> returning ",i)
    print(lst_str_2)
    return lst_str_2
  else:
    return make_lst_str_2(sets,i,lst_str_2)

def makecombinations(sets):
  if len(sets)!=1:
    lst_str = make_lst_str_set(sets[0],sets[1])
    ix=2
    if ix==len(sets):
      return lst_str
    else:
      lstout = make_lst_str_2(sets,ix,lst_str)
      print("--- getting the returned value")
      print("--- out from combinations: ", lstout)
      return lstout

def splitlist(lst):
  listout=[]
  for entry in lst:
    listout.append(entry.split("/"))
  return listout

def perform_plot_gridsearch(names,scores,keys,changeparameter,output_,suffix_):
  fig = plt.figure(figsize=(35,15))
  print("\n ----- perform_plot_gridsearch function -----")
  for name,score_obj,key,change in zip(names,scores,keys,changeparameter):
    print("\n\tname: ", name)#,end=" ")
    print(score_obj)
    print(score_obj.columns)
    #print(set(list(score_obj['param_max_depth'])))
  
    # get list of unique values for each variable 
    how_many_pars=len(key)
    par_set=[]
    for i_par in range(0,how_many_pars-1):
      if(key[i_par!=change]):
        par_set.append(set(list(score_obj[key[i_par]])))
    print(par_set)
    #listcomb = makecombinations(par_set)
    listcomb = splitlist(makecombinations(par_set))
    print(listcomb)

    # plotting a graph for every combination of paramater different from change (e.g.: n_estimator in random_forest): score vs. change
    pad = plt.subplot(1,len(names),names.index(name)+1)
    pad.set_title(name,fontsize=20)
    plt.ylim(-0.3,0)
    plt.xlabel(change,fontsize=15)
    plt.ylabel('neg_mean_squared_error',fontsize=15)
    pad.get_xaxis().set_tick_params(labelsize=15)
    pad.get_yaxis().set_tick_params(labelsize=15)

    for case in listcomb:
      df_case = score_obj.loc[(score_obj[key[0]]==float(case[0])) & (score_obj[key[1]]==float(case[1]))]
      lab="%s: %s, %s: %s"%(key[0],case[0],key[1],case[1])
      df_case.plot(x=change,y='mean_test_score',ax=pad,label=lab,marker='o')
      print(df_case)
    pad.legend(fontsize=20)
    
  plotname=output_+"/GridSearchResults"+name+suffix_+".png"
  plt.savefig(plotname)
  plt.show()
