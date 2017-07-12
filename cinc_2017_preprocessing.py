# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:21:23 2017
CinC 2017 analysis using BRT for HRT analysis and HRV
@author: obarquero, sandra cantero, rebeca goya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def change_Vpredict_Id(X_matrix):
    """
    Function that get the first digits before de hyphen in the Vpredict ID.  This
    is because sometimes de complete Vpredict-ID e.g 94-HM-007 does not match with 
    the same Vpredict-ID in other files. Mainly Holter-data vs clinical data does not
    match. However, the first digits before hyphen is a unique id
    """
    
    #load data
    #X_matrix = pd.read_csv(x_matrix_fname)
    
    #create new col Vpred_num_id
    X_matrix['Vpred_num_id'] = ['']*len(X_matrix)
    
    for i in range(len(X_matrix)):
        
        vp = X_matrix.Vpredict_Id[i]
        
        idx_hyphen = vp.index('-')
        
        vp_num_id = vp[:idx_hyphen]
        
        X_matrix.ix[i,'Vpred_num_id'] = int(vp_num_id)
                   
    return X_matrix
    
def get_rhythm(case_data,case,rhythm_data):
    """
    Function to obatin the rhythm from rhythm data using case data and a given id
    """
    idx = np.where(case_data['id']==case)[0][0]
    
    #get id to match the other dataframe
    id_vpred = case_data['id'][idx]
    
    #get the rhythm from rhythm dataframe
    
    idx_2 = np.where(rhythm_data['id']==id_vpred)[0][0]
    rhythm = rhythm_data['alta_paciente_ritmo'][idx_2]
    
    return rhythm
def get_cause(case_data,case,clinical_data):
    """
    Function to obatin the reason of admission from clinical dataa using case data and a given id
    """
    idx = np.where(case_data['id']==case)[0][0]
    
    #get id to match the other dataframe
    id_vpred = case_data['id'][idx]
    
    #get the rhythm from rhythm dataframe
    
    idx_2 = np.where(clinical_data['id']==id_vpred)[0][0]
    admin_cause = clinical_data['motivo_ingreso'][idx_2]
    
    return admin_cause
        
def add_sex_age_rhythm_clinical(x_matrix_fname,case_data_fn,alta_holter_fn,clinical_data_fn):
    """
    Function that adds the sex and age for each patient in X_matrix from Vpredict
    It also add the rhythm: sinusal, af,mp. In the study we are going to remove
    AF and MP rhythms. It also include clinical data (cause of inclusion), mainly:
    iam, descompensation
    It uses the clinical excel data set and also the X_matrix.
    """
    
    #read x_matrix data
    X_matrix = pd.read_csv(x_matrix_fname)
    
    #add new column id to get positive identificacion among clinical data with csv's
    
    X_matrix = change_Vpredict_Id(X_matrix)
    
    #read case data from csv
    case_data = pd.read_csv(case_data_fn,sep = ";")
    
    #read holter rhythm
    rhythm_data = pd.read_csv(alta_holter_fn,sep = ";")
    
    #read clinical data
    clinical_data = pd.read_csv(clinical_data_fn,sep = ";")
    
    #include new col for sex
    #0 for male
    #1 for female
    #include new col for age
    X_matrix['sex'] = [0]*len(X_matrix)
    X_matrix['age'] = [0]*len(X_matrix)
    
    #column for holter rhythm
    #sinusal
    #atrial_fibrillation
    #pace
    X_matrix['holter_rhythm'] = ['']*len(X_matrix)
    
    #Reason for admission
    #nan: unknown
    #iam: myocardial infarction
    #descompensation
    X_matrix['admission_cause'] = ['']*len(X_matrix)
    iii = 0
    for case in case_data['id']:
        print iii,'de',len(case_data)
        iii = iii + 1
        if np.sum(X_matrix['Vpred_num_id'] == case) == 0:
            continue
        else:
            idx = np.where(case_data['id']==case)[0][0]
            sex = case_data['sexo'][idx]
            age = case_data['edad'][idx]
            rhythm = get_rhythm(case_data,case,rhythm_data)
            admin_cause = get_cause(case_data,case,clinical_data)
            #print 'sexo',sex
            #print 'edad',age
            #print 'rhythm',rhythm
            #print 'admin cause',admin_cause
            
            #fill in sex
            if sex == 'femenino':
                #print 'f'
                X_matrix.ix[X_matrix['Vpred_num_id'] == case,'sex'] = 1
            elif sex == 'masculino':
                #print 'm'
                X_matrix.ix[X_matrix['Vpred_num_id'] == case,'sex'] = 0
            #fill in  age   
                          
            X_matrix.ix[X_matrix['Vpred_num_id']==case,'age'] = age
            
            #fill in rhythm

            X_matrix.ix[X_matrix['Vpred_num_id']==case,'holter_rhythm'] = rhythm
            
            #fill in admission cause
            
            X_matrix.ix[X_matrix['Vpred_num_id']==case,'admission_cause'] = admin_cause
                    
    
    #include age
    return X_matrix   


#TO DO

#limit the possible values of TS between -30 and 33. 

#TO DO

#verify why we obtain these extreme values

