# -*- coding: utf-8 -*-
"""
Create a X matrix with all the values extracted from a 24-hour Holter recording,
and a Y_TS, Y_TO matrixes with the TS and TO values calculated respectively.
"""

from __future__ import unicode_literals
import numpy as np
from HolterData import HolterData
from HRT import HRT
from HRV import HRV
import os
import glob
import scipy as sp
import matplotlib.pyplot as plt



class MatrixData(object):
    
    
    def __init__(self):
        self.X = np.array([])           #X matrix for one patient
        self.y_TO = np.array([])        #y_TO matrix for one patient
        self.y_TS = np.array([])        #y_TS matrix for one patient
        self.X_all = np.array([])       #X matrix for all patients
        self.y_TO_all = np.array([])    #y_TO matrix for all patients
        self.y_TS_all = np.array([])    #y_TS matrix for all patients
        self.ident = 0;
        self.labels = []
        self.sequence_list = []
        
        
        
    def get_final_matrix(self, ruta):
        """
        Create a X matrix with all the values extracted from all 24-hour Holter recording files
        associated to each patient included in the path, one below one other, 
        and their correspondent Y_TS, Y_TO matrixes with the TS and TO values calculated.
        """
        os.chdir(ruta) #ruta= 'data/panda'
   
        patients = []
        for file in glob.glob("*.txt"):
            patients.append(file)
            
        print patients            
            
        for i,pat in enumerate(patients):
            print i,'de',len(patients)
            print pat
            print "......."
            #We rewrite X, y_TS, y_TO for the next patient
            item = self.get_matrix_for_patient(pat,i+1) 
            if len(item)>0:
                self.sequence_list.append(item)    
                
                
            if i == 0:
                
                self.X_all = np.array(self.X)
                self.y_TS_all = np.array(self.y_TS)
                self.y_TO_all = np.array(self.y_TO)
            
            else:
                
                self.X_all = np.concatenate((self.X_all,np.array(self.X)))
                self.y_TS_all = np.concatenate((self.y_TS_all,np.array(self.y_TS)))        
                self.y_TO_all = np.concatenate((self.y_TO_all,np.array(self.y_TO))) 
                
        return self.X_all, self.y_TS_all, self.y_TO_all, self.labels, self.sequence_list
        
        
        
    def get_matrix_for_patient(self, ruta, ident):
        """
        Create a X matrix with all the values extracted from a 24-hour Holter recording associate to
        one patient, and their correspondent Y_TS, Y_TO matrixes with the TS and TO values calculated.
        """
        
        rr_3min_valid = []
        rr_3min_valid_corrected = []
        lab_3min_valid = []
        pos_3min_valid = []
        tachs_all_cond_valid = []
        V_pos_tachs_all_cond_valid = [] 
        name = ruta.split('.')[0]
        self.ident = ident
        
        #Extract the data from the file
        hd = HolterData()
        pat = hd.read_holter_file(ruta)
        
        #We saved the RR intervals and the labels of each one
        rr = pat['RRInt']
        labels = pat['Labels']
           
        #We calculate the parameters of Heart Turbulence 
        print "HRT"
        hrt = HRT(rr, labels)
        hrt_pat = hrt.fill_HRT()
        #We saved the valid tachogramas and their positions
        tachs_all_cond = hrt_pat['tachograms_ok']
        V_pos_tachs_all_cond = hrt_pat['v_pos_tachs_ok']
        
        #And with the periods of 3 minutes before each tachogram along with their positions
        rr_3min_all = hrt_pat['RR_before_V']
        pos_rr_3min_all = hrt_pat['pos_RR_bef_V']    

        #Of all the segments of 3 minutes, we only want those corresponding to the valid tachograms
        #To do this we go through all 3-min segments of RR-intervals before each VPC,
        #and we keep the valid tachograms already corrected (with their corresponding positions and associated labels)
        #and their corresponding 3-min segment of RR previous(with their corresponding positions and associated labels)
        ii = 0
        hrv = HRV()
        for rr_3min,pos_3min,tach,V_pos_tach in zip(rr_3min_all,pos_rr_3min_all,tachs_all_cond,V_pos_tachs_all_cond):
            
           # print ii," de ",len(rr_3min_all)
            
            ii = ii + 1
            
            lab_3min = labels[pos_3min[0]:pos_3min[1]] #get the labels for the actual 3 min seg
            hrv = HRV()
            ind_not_N_beats = hrv.artifact_ectopic_detection(rr_3min, lab_3min, 0.2)
            
            if hrv.is_valid(ind_not_N_beats):
                
                #correction
                rr_corrected = hrv.artifact_ectopic_correction(rr_3min, ind_not_N_beats)
                
                rr_3min_valid.append(rr_3min)
                rr_3min_valid_corrected.append(rr_corrected)
                
                lab_3min_valid.append(lab_3min)
                pos_3min_valid.append(pos_3min)
                
                tachs_all_cond_valid.append(tach)
                V_pos_tachs_all_cond_valid.append(V_pos_tach)
                
                
        #Once we have the 3-minute segments of RR for the corrected tachograms, 
        #we calculate the HRV variables for those segments
        print "HRV corrected ok. Start HRV indexes computing"
        hrv_pat = hrv.load_HRV_variables(rr_3min_valid_corrected)
        
        
        #We join the RR segment of 3 minutes together with the corrected valid tachogram associated for each 
        #and save it in a list
        list_rr3_and_tach = []
        for i in range(len(rr_3min_valid_corrected)):
            list_rr3_and_tach.append(list(rr_3min_valid_corrected[i]) + list(tachs_all_cond_valid[i]))
            

        
        #get the TS,TO,for the valid tachograms according to HRV filtering criteria   
        ts = []
        to = []
        
        for v_pos in hrt_pat['v_pos_tachs_ok']:
            if v_pos in V_pos_tachs_all_cond_valid:
                idx = hrt_pat['v_pos_tachs_ok'].index(v_pos)
                ts.append(hrt_pat['TS'][idx])
                to.append(hrt_pat['TO'][idx])
        
        #get CI and CP and SCL or the valid tachograms according to HRV filtering criteria        
        ci = []  
        cp = []
        scl = []
        
        for tach in tachs_all_cond_valid:
            ci.append(tach[5])
            cp.append(tach[6])
            scl.append(np.mean(tach[:5]))
            
        #Get y_TS and y_TO matrix    
        self.y_TS = np.array(ts)
        self.y_TO = np.array(to)
        
        #We save the name associated with each column in labels
        etiquetas = [ident, name]
        for elem in hrv_pat.keys():
            etiquetas.append(elem)
        others = ['scl','ci','cp','ts','to','ts_a','to_a']
        for elem in others:
            etiquetas.append(elem)
        
        #Get X matrix 
        X = np.array([hrv_pat[l] for l in hrv_pat.keys()]).T
        
            
        #We repeat the unique values
        idents = [ident]*len(scl)
        names = [name]*len(scl)
        ts_a = [hrt_pat['TS_average']]*len(scl)
        to_a = [hrt_pat['TO_average']]*len(scl)
        
        #Converted to vector column
        idents_array = np.array(idents)[:,np.newaxis]
        names_array = np.array(names)[:,np.newaxis]
        scl_array = np.array(scl)[:,np.newaxis]
        ci_array = np.array(ci)[:,np.newaxis]
        cp_array = np.array(cp)[:,np.newaxis]
        to_array = np.array(to)[:,np.newaxis]
        ts_array = np.array(ts)[:,np.newaxis]
        to_a_array = np.array(to_a)[:,np.newaxis]
        ts_a_array = np.array(ts_a)[:,np.newaxis]
        
        self.X = np.concatenate((idents_array, names_array, X, scl_array, ci_array, cp_array, to_array, ts_array, to_a_array, ts_a_array),axis= 1)        
        self.labels = etiquetas
        
        return list_rr3_and_tach

################################################################# 
############################  MAIN  #############################    
################################################################# 
    
#mD = MatrixData()    
#X, y_TS, y_TO, labels, sequences = mD.get_final_matrix('data/panda')


plt.close('all') 
a = sequences[0][15]   
x = range(len(a))
plt.figure()
plt.title("Segment of 3 minutes of RR-intervals followed by the VPC")
plt.xlabel("RR-interval (ms)")
plt.xlabel("Number of RR-interval")
plt.plot(x, a)


a = sequences[8][2]   
x = range(len(a))
plt.figure()
plt.title("Segment of 3 minutes of RR-intervals followed by the VPC")
plt.xlabel("RR-interval (ms)")
plt.xlabel("Number of RR-interval")
plt.plot(x, a) 

a = sequences[17][8]   
x = range(len(a))
plt.figure()
plt.title("Segment of 3 minutes of RR-intervals followed by the VPC")
plt.xlabel("RR-interval (ms)")
plt.xlabel("Number of RR-interval")
plt.xlim(0, 305)
plt.plot(x, a) 

