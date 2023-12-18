# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:06:30 2022

@author: aciarlo@unisa.it

"""

import numpy as np
from expyriment_stash.extras.expyriment_io_extras import tbvnetworkinterface
import os, pickle
import json
from sklearn.manifold import MDS


class rtRSA_data():
    
    def __init__(self, sub_name):
        
        """
        Initialization of the class. 
        
        Parameters:
        ----------------------------------------------------------------------
        sub_name: sting
            subject ID
                
        """
        self.name = sub_name
        #name of the base stimuli
        self.conditions = []
        #t-values of the base stimuli (NrOfBaseStimuli x NrOfCoordinates )
        self.base_patterns = []
        #coordinates of the ROI in the functional native space (FMR) from 
        #which the t-values of the base stimuli are extracted
        self.func_coords = []
        #dimension of the functional data (needed for voxels selection with searchlight)
        self.vol_dim = []
        #model RDM for searchlight
        self.modelRDM = np.loadtxt(os.getcwd()+'/modelRDM.mat')
        #searchlight config options
        self.searchlight_config = json.load(open(os.getcwd()+'/searchlight.cfg'))
        #selected coordinates after searchlight
        self.sel_func_coords =[]
        #t-values of the base stimuli after voxels selection (NrOfBaseStimuli x NrOfCoordinates)
        self.sel_base_patterns = []
        #representational dissimilarity matrix
        self.RDM = []
        #coordinates of the RS
        self.RS_coords = []
      
    def save(self,path):
        
        #save RDM
        np.savetxt(path +'/RDM.mat',self.RDM)
        
        #save RS coordinates
        np.savetxt(path +'/RS_coords.rcd',self.RS_coords)
        
        #save t values of the base patterns
        np.savetxt(path+'/base_patterns.tvals',self.base_patterns)
        np.savetxt(path+'/sel_base_patterns.tvals',self.sel_base_patterns)
                 
        #save names of the base stimuli
        f=open(os.path.join(path+'/stimuli.cnd'),'w')
        [f.write('%s\n' % item) for item in self.conditions]
             
        #save functional coordinates
        np.savetxt(path+'/func_coord.fnc',self.func_coords,fmt='%i')
        np.savetxt(path+'/sel_func_coord.fnc',self.sel_func_coords,fmt='%i')

        #save searchlight settings
        with open(path+'/searchlight.cfg','w') as f:
            json.dump(self.searchlight_config,f,indent=4)
        #save RDM
        np.savetxt(path +'/modelRDM.mat',self.modelRDM)
        
        #save subject info
        sub_data ={'sub_name':self.name,
                   'vol_dim':self.vol_dim}
        with open(path+'/sub.info','w') as f:
            json.dump(sub_data,f,indent=4)
        
        return self
        
    def load(self,path):
        
        sub_data = json.load(open(path+'/sub.info'))
        self.name = sub_data['sub_name']
        
        with open(path+'/stimuli.cnd','r') as f:
            self.conditions = [line.rstrip() for line in f.readlines()]
       
        self.base_patterns = list(np.loadtxt(path+'/base_patterns.tvals'))
        self.func_coords = np.loadtxt(path+'/func_coord.fnc').astype(int)
        self.vol_dim = sub_data['vol_dim']  
        self.modelRDM = np.loadtxt(path+'/modelRDM.mat')
        self.searchlight_config = json.load(open(path+'/searchlight.cfg'))
        self.sel_func_coords = np.loadtxt(path+'/sel_func_coord.fnc').astype(int)
        self.sel_base_patterns = list(np.loadtxt(path+'/sel_base_patterns.tvals'))
        self.RDM = np.loadtxt(path +'/RDM.mat')
        self.RS_coords = np.loadtxt(path+'/RS_coords.rcd')
        
        return self
    
    def sort_conditions(self):

        '''
        Sort conditions (and related patterns) in alphabetical order
        '''
        
        idx_sort = np.argsort(self.conditions)
        self.conditions = [self.conditions[idx] for idx in idx_sort]
        self.base_patterns = [self.base_patterns[idx] for idx in idx_sort]
        
    
    def mds(self):

            """                                                                                       
            Modern multidimensional scaling (MDS)                                                  
                                                                                                       
            Parameters                                                                                
            ----------                                                                                
            self.RDM : (n, n) array                                                                          
                Symmetric distance matrix.
        
            self.n_comp : int
                Number of components requested
                                                                    
                                                                                                       
            Returns                                                                                   
            -------                                                                                   
            self.RS_coords : (n, p) array                                                                          
                Configuration matrix. Each column represents a dimension. Only the                    
                p dimensions corresponding to positive eigenvalues of B are returned.                 
                Note that each dimension is only determined up to an overall sign,                    
                corresponding to a reflection.   
    
            self.RS_inv_mat : None                                                                            
                This output is not given by this scaling. The future projections
                will happen via optimization procedure.                                                     
                                                                                                                                                                     
                                                                                                       
            """
            
            #create the embedding for n components
            mds = MDS(dissimilarity='precomputed',
                      n_components=2, random_state=0)
            #transofrm the RDM in the embedding space
            #the algotrithm is initialized with the coordinates found with the 
            #classical mds
            self.RS_coords = mds.fit_transform(self.RDM)#, init=self.cmdscale())

        
            
        

        

        

        
   

        
       
        
