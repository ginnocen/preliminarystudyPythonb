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


def getclassifiers():
  classifiers = [GradientBoostingClassifier(learning_rate=0.01, n_estimators=2500, max_depth=1),
                    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                    AdaBoostClassifier(),DecisionTreeClassifier(max_depth=5)]
                  
  names = ["GradientBoostingClassifier","Random_Forest","AdaBoost","Decision_Tree"]
  return classifiers, names

def getvariablestraining():
  mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]
  return mylistvariables

def getvariablesothers():
  mylistvariablesothers=['inv_mass_ML','pt_cand_ML']
  return mylistvariablesothers

def getvariableissignal():
  myvariablesy='signal_ML'
  return myvariablesy

def getvariablesall():
  mylistvariables=getvariablestraining()
  mylistvariablesothers=getvariablesothers
  myvariablesy=getvariableissignal()
  return mylistvariablesall

def preparestringforuproot(myarray):
  arrayfinal=[]
  for str in myarray:
    arrayfinal.append(str+"*")
  return arrayfinal
    
def fit(names_, classifiers_,X_train_,y_train_):
  trainedmodels_=[]
  for name, clf in zip(names_, classifiers_):
    clf.fit(X_train_, y_train_)
    trainedmodels_.append(clf)
  return trainedmodels_

def test(names_,trainedmodels_,X_test_,test_set_):
  for name, model in zip(names_, trainedmodels_):
    y_test_prediction=[]
    y_test_prob=[]
    y_test_prediction=model.predict(X_test_)
    y_test_prob=model.predict_proba(X_test_)[:,1]
    test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)
    test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
  return test_set_

def savemodels(names_,trainedmodels_,folder_,suffix_):
  for name, model in zip(names_, trainedmodels_):
    fileoutmodel = folder_+"/"+name+suffix_+".sav"
    pickle.dump(model, open(fileoutmodel, 'wb'))

def readmodels(names_,folder_,suffix_):
  trainedmodels_=[]
  for name in names_:
    fileinput = folder_+"/"+name+suffix_+".sav"
    model = pickle.load(open(fileinput, 'rb'))
    trainedmodels_.append(model)
  return trainedmodels_



def importanceplotall(mylistvariables_,names_,trainedmodels_,suffix_):
  figure1 = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, model in zip(names_, trainedmodels_):
    ax = plt.subplot(2, len(names_)/2, i)  
    #plt.subplots_adjust(left=0.3, right=0.9)
    feature_importances_ = model.feature_importances_
    y_pos = np.arange(len(mylistvariables_))
    ax.barh(y_pos, feature_importances_, align='center',color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mylistvariables_, fontsize=17)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance',fontsize=17)
    ax.set_title('Importance features '+name, fontsize=17)
    ax.xaxis.set_tick_params(labelsize=17)
    plt.xlim(0, 0.7)
    i += 1
  plotname='plots/importanceplotall%s.png' % (suffix_)
  plt.savefig(plotname)
