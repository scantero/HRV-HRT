# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:28:32 2017

@author: obarquero

Created on Thu Apr 13 09:53:44 2017
HRT analysis Cinc 2017.
1.- Remove patients with fa an mp
2.- Separate data between iam an descompesacion patients
3.- Train using this split.
4.- Separete using TS > 2.5 && TO < 0 (verify the number of patients in each group)
otherwise use TS > 2.5
@author: obarquero
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
#from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.datasets.california_housing import fetch_california_housing
from cinc_2017_preprocessing import *
import numpy as np
import pandas as pd


#%%
def add_sex_age_rhythm_clinical(x_matrix_fname,clinical_data_fn):
    """
    Function that adds the sex and age for each patient in X_matrix from Vpredict
    It also add the rhythm: sinusal, af,mp. In the study we are going to remove
    AF and MP rhythms. It also include clinical data (cause of inclusion), mainly:
    iam, descompensation
    It uses the clinical excel data set and also the X_matrix.
    """
    
    #read x_matrix panda using np
    X_matrix = np.load('array_x_data.npy')
    
    
    #add new column id to get positive identificacion among clinical data with csv's
    clinical_data = pd.read_csv(clinical_data_fn,sep = ";")
    
    
    #read case data from csv
    
    #include new col for sex #0 for male #1 for female
    #include new col for age
    sex = np.zeros((len(X_matrix),1)) -1000
    sex[:] = np.nan
    age = np.zeros((len(X_matrix),1)) -1000
    age[:] = np.nan
    
    aaa = 0
    
    for c,row in enumerate(X_matrix):        
        
        #get id using the fact that is from start to the last position of '_'
        idx_id = [i for i, ltr in enumerate(row[1]) if ltr == '_']
        id_str = row[1][0:idx_id[-1]]
    
        if sum(clinical_data['ID']==id_str) == 0: 

            #print clinical_data['ID'] + ' id extraido ' + id_str
            continue
        
        sex[c] = clinical_data[clinical_data['ID']==id_str]['Sexo']
        age[c] = clinical_data[clinical_data['ID']==id_str]['EDAD']
                
    
    #include age and sex intro X_matrix
    X_matrix = np.hstack((X_matrix,sex,age))    
    
    return X_matrix   


#TO DO

#limit the possible values of TS between -30 and 33. 

#TO DO

#verify why we obtain these extreme values
#%%
#get data with all the columns
x_matrix_fname = 'array_x_data.npy'
clinical_data_fn = "panda_clinical_data.csv"
X_matrix_2 = add_sex_age_rhythm_clinical(x_matrix_fname,clinical_data_fn)

#remove -1000 in sex and age



#%%

features = ['Id','panda_Id','SDNN','TriangIndex','SDSD','PNN50','TINN','AVNN','LogIndex','RMSSD','NN50','SCL','CI','CP','TO','TS','TO_a','TO_a','Sex','Age'] #sex= 0 male, sex= 1 female
#3. Selec the features to train and train BRT on each subset

id_X_feat = [2,3,4,6,7,8,9,11,12,13,18,19] #without, pnn50, nn50
feats = np.asarray(features)[id_X_feat]

#%%
#Get X and y

#Split X_matrix_2 into three diff matrix into
#split intro HRT normal and non-normal
#In most clinical studies, however, TO 0% and TS 2.5 ms/R-R interval are considered normal.
#These originally proposed cutoff values were validated in the data of 3 large
#post-infarction studies (totaling 2,646 patients) (1,92).
#The TO and TS variables can be used as separate clinical
#variables or in a combination. In risk stratification studies
#(see the subsequent section), HRT values are usually classified
#into 3 categories: 1) HRT category 0 means TO and
#TS are normal; 2) HRT category 1 means 1 of TO xor TS is
#abnormal; and 3) HRT category 2 means both TO and TS
#are abnormal. 

y_ts_a = X_matrix_2[:,17].astype('float32') #id = 17
y_to_a = X_matrix_2[:,16].astype('float32') #id = 16

id_hrt_1 = np.logical_and(y_ts_a  > 2.5, y_to_a <  0)
id_hrt_2 = np.logical_xor(y_ts_a <= 2.5, y_to_a >= 0)
id_hrt_3 = np.logical_and(y_ts_a <= 2.5, y_to_a >= 0)
#get cases with HRT normal

X_a_1 = X_matrix_2[id_hrt_1,:]
X_a_2 = X_matrix_2[id_hrt_2,:]
X_a_3 = X_matrix_2[id_hrt_3,:]

X_a_1 = X_a_1[:,id_X_feat].astype('float32')
X_a_2 = X_a_2[:,id_X_feat].astype('float32')
X_a_3 = X_a_3[:,id_X_feat].astype('float32')

#remove cases with any NaN
X_hrt_1 = np.delete(X_a_1,np.where(np.isnan(X_a_1))[0],axis = 0)
X_hrt_2 = np.delete(X_a_2,np.where(np.isnan(X_a_2))[0],axis = 0)
X_hrt_3 = np.delete(X_a_3,np.where(np.isnan(X_a_3))[0],axis = 0)

y_ts_1 = X_matrix_2[id_hrt_1,15].astype('float32')
y_ts_1 = np.delete(y_ts_1,np.where(np.isnan(X_a_1))[0],axis = 0)
y_to_1 = X_matrix_2[id_hrt_1,14].astype('float32')
y_to_1 = np.delete(y_to_1,np.where(np.isnan(X_a_1))[0],axis = 0)

y_ts_2 = X_matrix_2[id_hrt_2,15].astype('float32')
y_ts_2 = np.delete(y_ts_2,np.where(np.isnan(X_a_2))[0],axis = 0)
y_to_2 = X_matrix_2[id_hrt_2,14].astype('float32')
y_to_2 = np.delete(y_to_2,np.where(np.isnan(X_a_2))[0],axis = 0)

y_ts_3 = X_matrix_2[id_hrt_3,15].astype('float32')
y_ts_3 = np.delete(y_ts_3,np.where(np.isnan(X_a_3))[0],axis = 0)
y_to_3 = X_matrix_2[id_hrt_3,14].astype('float32')
y_to_3 = np.delete(y_to_3,np.where(np.isnan(X_a_3))[0],axis = 0)


#xxxxxxxxxxxxxxxx

#%%
#train/test
import matplotlib.pyplot as plt

def brt_train_plot_func(X,y,feats,train_model = False):
    """
    Function to train the BRT model and plots to make inference
    """

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=1)
    
    #y_train = y_train.as_matrix().ravel()
    #y_test = y_test.as_matrix().ravel() using only when coming from pandas dataframe
    
    param_grid_brt = {'learning_rate': np.logspace(-4,0,50),'max_depth': range(2,8),'min_samples_leaf': range(3,10)}
    #from sklearn.metrics import mean_squared_error, make_scorer
    #param_grid_brt = {'learning_rate': np.logspace(-4,0,3),'max_depth': [2,6],'min_samples_leaf': [3,10]}
    clf = GradientBoostingRegressor(n_estimators=500)   
    #cross-validation grid to search the best parameters
    
    
    #str_in = raw_input("(T)raining or (U)sed selected (Default: U): ")
    
    if train_model:
        print "Training model"
        #mse_scorer = make_scorer(mean_squared_error,greater_is_better = False)
        brt_complete = GridSearchCV(clf, param_grid_brt,n_jobs = -1,verbose = True,cv = 10)
        brt_complete.fit(X_train,y_train)
        brt = brt_complete.best_estimator_
    else:
        brt = GradientBoostingRegressor(n_estimators=2000,learning_rate=0.0008,max_depth = 4,min_samples_leaf=5)
        brt.fit(X_train,y_train)
        
    #str_in = raw_input("Descomp-(T)raining or (U)sed selected (Default: U): ")
    #
    #if str_in == 'T':
    #    print "Training descomp model"
    #    brt_descomp_complete = GridSearchCV(clf_descomp, param_grid_brt,n_jobs = -1,verbose = True,cv = 10)
    #    brt_descomp_complete.fit(X_descomp_train,y_descomp_train)
    #    brt_descomp = brt_descomp_complete.best_estimator_
    #else:
    #    brt_descomp = GradientBoostingRegressor(n_estimators=2000,learning_rate=0.006,max_depth = 4,min_samples_leaf=5)
    #    brt_descomp.fit(X_descomp_train,y_descomp_train)
    
    
    plt.close('all')
    #  ####### IAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #relative importance
    
    feature_importance = brt.feature_importances_
        # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())     
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    #plt.sca(axs[5])
    #plt.cla()
    #feats = np.array(features)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos,feats[sorted_idx], fontsize=20)  
    plt.title("TS Group 3", fontsize=20)
    plt.xlabel('Relative Importance (%)', fontsize=20)
    plt.subplots_adjust(top=0.9, left=0.18, bottom=0.15)
    #partial dependence plot
    
    #mse
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_pred = brt.predict(X_test)
    
    print "MSE",mean_squared_error(y_test,y_pred)
    print 'R2',r2_score(y_test,y_pred)
    
    #plot for IAM
    #plt.figure()
    #4 features AVNN, age, sex, ci
    #features = ['SDNN','HRVTriangIndex','SDSD','AVNN','logIndex','RMSSD','ci','sex','age']
    #target_features = [features[3],features[-1],features[-2],features[-3]]
    target_features_idx = [0,4,7,3,9,(0,4)]
    fig_hrt, axs = plot_partial_dependence(brt, X_train, target_features_idx, feature_names=feats,n_jobs=-1, grid_resolution=80)
    fig_hrt.suptitle('TS Group 3 = f(HRV)', fontsize=20)
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.5)
    for a in range(5):
        axs[a].set_ylabel("TS", fontsize=20)  # tight_layout causes overlap with suptitle
        axs[a].set_xlabel(feats[target_features_idx[a]], fontsize=20)
    axs[5].set_xlabel(feats[target_features_idx[5][0]], fontsize=20)    
    axs[5].set_ylabel(feats[target_features_idx[5][1]], fontsize=20)
    plt.show()
    
    target_features_idx = [8,7]
    fig_hrt, axs = plot_partial_dependence(brt, X_train, target_features_idx, feature_names=feats,n_jobs=-1, grid_resolution=80)
    fig_hrt.suptitle('TS Group 3 = f(HRV)', fontsize=20)
    plt.subplots_adjust(top=0.9, left=0.12)
    for a in range(2):
        axs[a].set_ylabel("TS partial dependence", fontsize=20)  # tight_layout causes overlap with suptitle
        axs[a].set_xlabel(feats[target_features_idx[a]], fontsize=20)
        axs[a].set_ylim(-2.5,1.5) 
    plt.show()
    
    
    fig = plt.figure()
        
    target_feature = (7, 3)
    pdp, (x_axis, y_axis) = partial_dependence(brt, target_feature, X=X_train, grid_resolution=80)
    XX, YY = np.meshgrid(x_axis, y_axis)
    Z = pdp.T.reshape(XX.shape).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
    ax.set_xlabel(feats[target_feature[0]], fontsize=18)
    ax.set_ylabel(feats[target_feature[1]], fontsize=18)
    ax.set_zlabel('$TS$', fontsize=18)
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('$TS = f(Scl,TINN)$', fontsize=18)
    #plt.subplots_adjust(top=0.91)


#
#%% 
# Separating according to HRT TS >2.5 and TO <0

brt_train_plot_func(X_hrt_3, y_ts_3, feats)


