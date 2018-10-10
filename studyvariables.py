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

#__________________________________________________________________________________
def CheckDir(path):
        if not os.path.exists(path):
                os.makedirs(path)
#__________________________________________________________________________________
def ProgressBar(part,tot):
        sys.stdout.flush()      # deletes what dumped in the line
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
def DoCorrMatrix(dataframe,number,string,c_type="pearson"):
        figCorr = plt.figure(num=number,figsize=(15,15))
        figCorr.suptitle("%s correlation - %s"%(c_type,string),fontsize=40)
        padcorr = plt.subplot(1,1,1)
        axis = padcorr.matshow(dataframe.corr(c_type))
        figCorr.colorbar(axis)
        padcorr.xaxis.set_ticks(range(0,len(dataframe.columns)))
        padcorr.yaxis.set_ticks(range(0,len(dataframe.columns)))
        padcorr.xaxis.set_ticks_position('bottom')
        padcorr.set_xticklabels(['']+dataframe.columns,rotation=45)#,fontsize='x-small')
        padcorr.set_yticklabels(['']+dataframe.columns)#,fontsize='x-small')
        plt.savefig("CorrMatrix%s_%s.pdf"%(c_type,string))
#__________________________________________________________________________________
def DimReduction(dataframe,varlist,n_pca,n_cases,name,dohistbool=True):

        data        = dataframe.loc[:,varlist]                          # get the dataframe with only the variables in 'varlist' 
        data_values = data.values                                       # get only the values
        data_values = StandardScaler().fit_transform(data_values)       # standardize all the values: gaussian distibution with mean=0 and sigma=1

        pca = PCA(n_pca)                                                # define the PCA object
        principalComponent = pca.fit_transform(data_values)
        
        pca_name_list = []
        for i_pca in range(1,n_pca+1):
                pca_name_list.append("Princ. comp. %d"%i_pca)

        pca_dataframe = pd.DataFrame(data=principalComponent,columns=pca_name_list)
        print(pca_dataframe.head(10))

        # do scatter plots on principal components
        pcaScatterFig = plt.figure(figsize=(15,15))
        pcaScatterFig.suptitle("Dim. reduction %s - %d principal components"%(name,n_pca),fontsize=40)
        index = 1
        rowscols = GetRowCol(n_cases)
        rows = rowscols[0]
        cols = rowscols[1]
        for i_col in range(1,n_pca+1):
                namex = pca_dataframe.columns[i_col-1]
                for i_col2 in range(1,n_pca+1):
                        if i_col2>i_col:
                                namey = pca_dataframe.columns[i_col2-1]
                                padSc = plt.subplot(rows,cols,index)
                                color=""
                                if name=="Signal":
                                        color+="red"
                                elif name=="Background":
                                        color="blue"
                                plt.scatter(pca_dataframe[namex],pca_dataframe[namey],s=3,c=color,marker="o",alpha=0.3)
                                padSc.set_xlabel(namex)
                                padSc.set_ylabel(namey)
                                index += 1
                                #padSc.set_title("Pearson corr. %f"%np.corrcoef(pca_dataframe[namex],pca_dataframe[namey])[0,1])
                        else:
                                continue
        plt.subplots_adjust(hspace=0.75,wspace=0.75)
        #plt.show()
        
        path="./%s/PCA"%name
        CheckDir(path)
        plt.savefig(path+"/%d.png"%n_pca, bbox_inches="tight")   
             
        print("\n--- ",pca.explained_variance_ratio_)
        if dohistbool>0:
                DoCorrMatrix(pca_dataframe,None,"pca_%s"%name)
                DoHist(pca.explained_variance_ratio_,name)
        #plt.show()
#__________________________________________________________________________________
def DoHist(array,name):
        figVarRat = plt.figure(figsize=(15,15))
        figVarRat.suptitle(name,fontsize=40)
        pad = plt.subplot(1,1,1)
        position = range(1,len(array)+1)
        col=""
        if name=="Signal":
                col+="red"
        elif name=="Background":
                col="blue"
        plt.bar(position,array,color=col)
        pad.set_xlabel("principal component",fontsize=20)
        pad.set_ylabel("explained variance ratio",fontsize=20)
        pad.set_ylim(0,0.3)
        pad.get_xaxis().set_tick_params(labelsize=20)
        pad.get_yaxis().set_tick_params(labelsize=20)
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



time0 = datetime.now()

# ----- input parameters management -----
print("===========================================================================")
print("\tCode started at ",time0.strftime('%d/%m/%Y, %H:%M:%S'))
print("\tList of arguments: ",str(sys.argv))
print("===========================================================================\n")
SaveScatterPlots = sys.argv[1]
print(SaveScatterPlots)
# ---------------------------------------

          
# variable list (NB: this is a sub-sample of variables stored in the DataFrame inside "testsample.pkl")  
# ===== variables to be checked for correlations =====
mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]    
   
train_set = pd.read_pickle("testsample.pkl")    # load objects stored in the file specified in the path 

print("\n - train_set data type:%s \n"% type(train_set))     # train_set is a pandas DataFrame
print(train_set)                                        # DataFrame dumped on the screen

print("\n... too confused... let's print the variables stored:")
print(train_set.columns)        # .columns returns an Index type with the indices of each column (i.e.: the name of variables stored)

############################################################
#   'signal_ML' is a colum of the DataFrame which flags:   #
#       - the SIGNAL candidates with 1.0                   #
#       - the BACKGROUND candidates with 0.0               #
############################################################
#   the function .los retrieves a column of the DataFrame, #
#   as a  DataFrame as well (Series)                       #
############################################################
train_set_sig=train_set.loc[train_set['signal_ML'] == 1]        # .loc function used to return a DataFrame with all the info about the entries satisfying the condition in brackets     
train_set_bkg=train_set.loc[train_set['signal_ML'] == 0]
print("\n --- train_set_sig type: %s\n --- train_set_bkg type: %s\n"%(type(train_set_sig),type(train_set_bkg)))         # train_set_sig and train_set_bkg are DataFrame (Series) as well

X_test= train_set_sig[mylistvariables]
num_col = len(X_test.columns)
print("\n=== A new DataFrame X_test, sub-DataFrame of train_set, has been created. The number of columns is %s. These are the variables stored in:"%num_col)
for i_labels in range(0,num_col):
        print("%s)\t%s"%(i_labels,X_test.columns[i_labels]))


figure = plt.figure(figsize=(18,15))
figure.suptitle('Variables distributions', fontsize=40)

#a1 = plt.subplot(4, 3, 1)                       # it means: the figure is subdivided in 4x3 subplots and I am selecting the number 1 (the first one! The numerations strarts from 1)
#plt.xlabel('d_len_xy_ML',fontsize=11)
#plt.ylabel("entries",fontsize=11)
#plt.yscale('log')
#kwargs = dict(alpha=0.3,density=True, bins=100)
#n, bins, patches = plt.hist(train_set_sig['d_len_xy_ML'], facecolor='r', label='signal', **kwargs)
#n, bins, patches = plt.hist(train_set_bkg['d_len_xy_ML'], facecolor='b', label='background', **kwargs)
#a1.legend()

#a2 = plt.subplot(4, 3, 2)                       
#name_var = train_set_sig.columns[1]
#plt.xlabel(name_var,fontsize=11)
#plt.ylabel("entries",fontsize=11)
#plt.yscale('log')
#kwargs = dict(alpha=0.3,density=True, bins=100)
#n, bins, patches = plt.hist(train_set_sig[name_var], facecolor='r', label='signal', **kwargs)
#n, bins, patches = plt.hist(train_set_bkg[name_var], facecolor='b', label='background', **kwargs)
#a2.legend()

print("\n==================================")
print("   Variable distribution plots")
print("==================================\n")
# drawing all the variables
num_variables = len(train_set_sig.columns)
for i_subpad in range(1,num_variables):                 # NB: the last variable is NOT PLOTTED 
        s_pad = plt.subplot(4, 3, i_subpad)             # it means: the figure is subdivided in 4x3 subplots and I am selecting the number i_subpad (the i_subpad th one! The numerations strarts from 1)        
        name_var = train_set_sig.columns[i_subpad-1]
        plt.xlabel(name_var,fontsize=11)                # equivalent to s_pad.set_xlabel
        plt.ylabel("entries",fontsize=11)               # equivalent to s_pad.set_ylabel
        plt.yscale('log')
        #####################################################################################################
        #       In this way the following dictionary is produced:                                           #
        #       {'alpha': 0.3, 'density': True, 'bins': 100}                                                #
        #       This is  a dictionary used to pass to the function some values as key-word parameters       #
        #####################################################################################################
        kwargs = dict(alpha=0.3,density=True, bins=100)
        n, bins, patches = plt.hist(train_set_sig[name_var], facecolor='r', label='signal', **kwargs)           # NB: this function returns MORE THAN ONE ARGUMENT
        n, bins, patches = plt.hist(train_set_bkg[name_var], facecolor='b', label='background', **kwargs)
        s_pad.legend()
plt.subplots_adjust(hspace=0.5)
#plt.show()     
plt.savefig("Var_distributions.pdf")#,bbox_inches='tight')


# check if there are the directories where to save the files, otherwise create them
path_Sig = "./Signal"
path_Bkg = "./Background"
CheckDir(path_Sig)
CheckDir(path_Bkg)


print("\n\n==================\n   kwargs: %s\n=================="%kwargs)

'''
# ------------------------------------------
#             Scatter_matrix
# ------------------------------------------

###########################################################################
#   the Pandas 'scatter_matrix' funtion does a multiple scatter plot.     #
#   It returns a 2D NumPy object that contains the combination nxn axis   #     ---> it is an array of pads, of which we can access the x and y axis!
#   (n axis for the n variables on x, n axis for the n variables on y)    #
###########################################################################
scplot_Sig = pd.scatter_matrix(train_set_sig,figsize=(16,9),alpha=0.3,s=1,c="red",marker=",")          # it does automatically the scatter plots of ALL the variables with the others ... messy plot!
scplot_Bkg = pd.scatter_matrix(train_set_bkg,figsize=(16,9),alpha=0.3,s=1,c="blue",marker=",")         # it does automatically the scatter plots of ALL the variables with the others ... messy plot!

print("--- scplot_Sig type: %s\t dim: %s\t shape: %s"%(type(scplot_Sig),scplot_Sig.ndim,scplot_Sig.shape))
#print(scplot_Sig)
print("--- scplot_Bkg type: %s\t dim: %s\t shape: %s"%(type(scplot_Bkg),scplot_Bkg.ndim,scplot_Bkg.shape))
#print(scplot_Bkg)

###############################################################################################
#   In order to access each single axis, the most comfortable way is to "flatten" the NumPy.  #
#   Without doing any copy, but just accessing it as it is flattened, we can use the ravel()  #
#   function ---> row of pads, of which we can access x and y axis                            #
#   With the ravel() function there is an ordered iteration among all the 'subpads'           #
###############################################################################################
for p_sig in scplot_Sig.ravel():
        p_sig.set_xlabel(p_sig.get_xlabel(),fontsize=10, labelpad=20)
        p_sig.set_ylabel(p_sig.get_ylabel(),fontsize=10, labelpad=50, rotation=0)
plt.savefig(path_Sig+"/matrix_Sig.png")

for p_bkg in scplot_Bkg.ravel():
        p_bkg.set_xlabel(p_bkg.get_xlabel(),fontsize=10, labelpad=20)
        p_bkg.set_ylabel(p_bkg.get_ylabel(),fontsize=10, labelpad=50, rotation=0)
plt.savefig(path_Bkg+"/matrix_Bkg.png")
#plt.show()
'''

# ------------------------------------------
#          Manual scatter plots
# ------------------------------------------
# let's do the scatter plots manually
print("\n==================================")
print("\tScatter plots")
print("==================================\n")

print("--- Different coefficients:")
print("\nPEARSON (signal):")
print(train_set_sig.corr('pearson'))
print("\nKENDALL (signal):")
print(train_set_sig.corr('kendall'))
print("\nSPEARMAN (signal):")
print(train_set_sig.corr('spearman'))

# --- correlation matrix SIGNAL ---

c_type = 'pearson'
#c_type = 'kendall'
#c_type = 'spearman'
DoCorrMatrix(train_set_sig,100,"signal",c_type)
DoCorrMatrix(train_set_bkg,101,"background",c_type)
'''
figCorr_sig = plt.figure(num=100,figsize=(15,15))
figCorr_sig.suptitle("%s correlation - signal"%c_type,fontsize=40)
padcorr_sig = plt.subplot(1,1,1)
axis_sig = padcorr_sig.matshow(train_set_sig.corr(c_type))
figCorr_sig.colorbar(axis_sig)
padcorr_sig.xaxis.set_ticks(range(0,len(train_set_sig.columns)))
padcorr_sig.yaxis.set_ticks(range(0,len(train_set_sig.columns)))
padcorr_sig.xaxis.set_ticks_position('bottom')
padcorr_sig.set_xticklabels(['']+train_set_sig.columns,rotation=90)
padcorr_sig.set_yticklabels(['']+train_set_sig.columns)
plt.savefig("corrmatrix_sig.png")
'''

#plt.show()

if SaveScatterPlots == "True":
        num_variables_corr = len(mylistvariables)
        for i_col in range(1,num_variables_corr):
                #namex = train_set_sig.columns[i_col-1] 
                namex = mylistvariables[i_col-1]              

                index_plot_sig = 1+i_col                # signal plots 
                index_plot_bkg = 1+i_col+num_variables_corr  # background plots

                fig_scPlot     = plt.figure(num=index_plot_sig,figsize=(18,15))       # signal        
                fig_scPlot.suptitle("Scatter plot SIGNAL %s"%namex,fontsize=40)
                fig_scPlot_BKG = plt.figure(num=index_plot_bkg,figsize=(18,15))       # background    
                fig_scPlot_BKG.suptitle("Scatter plot BACKGROUND %s"%namex,fontsize=40)    

                for i_col2 in range(1,num_variables_corr):               
                        namey = mylistvariables[i_col2-1]

                        plt.figure(index_plot_sig)   # moving to figure 1 (signal)
                        s_pad_scPlot = plt.subplot(3,3,i_col2)
                        plt.scatter(train_set_sig[namex],train_set_sig[namey],s=3,c="red",marker="o",alpha=0.3)
                        # ----- correlation coefficient -----
                        ndarray_corr_sig = np.corrcoef(train_set_sig[namex],train_set_sig[namey])
                        #print("\n=== Correlation %s vs. %s \n\tCorrelation coefficient matrix SIGNAL: \n%s"%(namex,namey,ndarray_corr_sig))  
                        s_pad_scPlot.set_title("Correlation coefficient: %.4f"%ndarray_corr_sig[0,1])
                        # -----------------------------------
                        s_pad_scPlot.set_xlabel(namex)
                        s_pad_scPlot.set_ylabel(namey)
                        plt.subplots_adjust(wspace=0.25, hspace=0.3)
                        plt.savefig(path_Sig+"/scatter_SIG_%d.png"%i_col, bbox_inches="tight")

                        plt.figure(index_plot_bkg)   # moving to figure 1 (background)
                        fig_scPlot_BKG = plt.subplot(3,3,i_col2)
                        plt.scatter(train_set_bkg[namex],train_set_bkg[namey],s=3,c="blue",marker="o",alpha=0.3)
                        # ----- correlation coefficient -----
                        ndarray_corr_bkg = np.corrcoef(train_set_bkg[namex],train_set_bkg[namey])
                        #print("\tCorrelation coefficient matrix BACKGROUND: \n%s"%ndarray_corr_bkg)  
                        fig_scPlot_BKG.set_title("Correlation coefficient: %.4f"%ndarray_corr_bkg[0,1])
                        # -----------------------------------
                        fig_scPlot_BKG.set_xlabel(namex)
                        fig_scPlot_BKG.set_ylabel(namey)
                        plt.subplots_adjust(wspace=0.25, hspace=0.3)
                        plt.savefig(path_Bkg+"/scatter_BKG_%d.png"%i_col, bbox_inches="tight")
                        ProgressBar((i_col-1)*(num_variables_corr-1)+i_col2,(num_variables_corr-1)**2)
                        #plt.show()
#plt.show()      # after calling 'show', the current plot and axis are destroyed (namely: calling plt.savefig you do not save anything!) ---> all the plots produced in the program are plotted all together!
'''
maxN = len(mylistvariables)
for n_pc in range(2,maxN+1):
        dohistboolean = 1
        if n_pc<maxN:
                dohistboolean = 0
        print("\n\n\n========= n_pc %s   dohistboolean %s"%(n_pc,dohistboolean))
        DimReduction(train_set_sig,mylistvariables,n_pc,NComb(n_pc),"Signal",dohistboolean)
        DimReduction(train_set_bkg,mylistvariables,n_pc,NComb(n_pc),"Background",dohistboolean)
'''
DimReduction(train_set_sig,mylistvariables,len(mylistvariables),NComb(len(mylistvariables)),"Signal")
DimReduction(train_set_bkg,mylistvariables,len(mylistvariables),NComb(len(mylistvariables)),"Background")

plt.show()



time1 = datetime.now()
howmuchtime = time1-time0
print("\n===\n===\tExecution END. Start time: %s\tEnd time: %s\t(%s)\n==="%(time0.strftime('%d/%m/%Y, %H:%M:%S'),time1.strftime('%d/%m/%Y, %H:%M:%S'),howmuchtime))
























