# -*- coding: utf-8 -*-

"""
Created on Tue Apr 19 14:43:31 2022

@author: aciarlo@unisa.it

rtRSA feedback creation example. 
This script uses both the old and the new projection procedure for feedback generation.

"""
import os, pickle, glob
import numpy as np
import utils.nfrsa as nfrsa
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from utils.new_triangulation import create_baseline_figure, create_feedback, optimise_projection
from scipy.stats import pearsonr
import math
import time


# working directory
datadir = f'{os.getcwd()}/example_data/'
subs = glob.glob(os.path.join(datadir,'sub*'))

# output variables

all_ED = []
prj_type = []
repeated_sub_list = []
repeated_session_list = []
orig_corr_distances = []

output = dict({'Subject':[],
                       'Run':[],
                       'ED-CD Correlation':[],
                       'ED-CD p-value Correlation':[],
                       'Projection type':[],
                       'FB_position':[],
                       'Success flag':[],
                       'Initial_position':[],
                       'Num iter' : [],
                       'Fun eval':[],
                       'Execution time':[]}) 



for idx_sub,sub in enumerate(subs):
    
    sessions = glob.glob(os.path.join(sub,'old_proj/*output*'))
    sessions.sort()
    
    for session in sessions:
        
        all_flag = []   # flag for success of the new projection method ('exact' or 'approx')
        all_x0 = []     # initial position of the minimisation procedure
        all_fb_positions = []   # all feedback positions for the NEW projection
        timer = []  # computation time of the NEW projection
        num_iter = []   # num of iterations of the NEW projection
        fun_vals =[]    # output value of the minimisation procedure

        sub = os.path.basename(sub)
        session = os.path.basename(session)

        print(f'Subject: {sub}\nRun: {session}')

        #define idx_trg
        if session.split('_')[0]=='dog':
            idx_trg = 0
        
        if session.split('_')[0] =='cat':
            idx_trg = 1
            
        if session.split('_')[0]=='hammer':
            idx_trg = 2
        
        if session.split('_')[0] =='chair':
            idx_trg = 3   


        ###----------- LOAD OLD PROJECTION DATA ------------###

        print('Loading OLD projection data ...')
            
        #pick the subject's json file for the OLD projection (the classical MDS was used for RS generation)
        sub_json_file = os.path.join(datadir,sub,'old_proj',sub+'.json')
        
        #empty RSA object
        rtRSAObj = nfrsa.rtRSA(' ',2,' ')
        
        #load properties in the rtRSAObj
        rtRSAObj.load(sub_json_file)
        
        
        #load the t-values for each feedback block (previously extracted during the NF experiment)

        nf_run_tvalues = pickle.load(open(os.path.join(datadir,
                                                    sub,
                                                    'old_proj',
                                                    session,
                                                    'all_tvalues.pkl'),'rb'))
    
        
        old_rsc = rtRSAObj.RS_coords
        old_rdm = rtRSAObj.RDM
        
        
        #load the correlation distances between the NF patterns and the base patterns (previously computed during the NF experiment)
        fb_distances = np.array(pickle.load(open(os.path.join(datadir,
                                                     sub,
                                                     'old_proj',
                                                     session,
                                                     'feedback_distances.pkl'),'rb')))   

        nr_of_trials = fb_distances.shape[0]

        #old feedback positions
        old_fb_pos = np.array(pickle.load(open(os.path.join(datadir,
                                                     sub,
                                                     'old_proj',
                                                     session,
                                                     'feedback_positions.pkl'),'rb')))
        


        # compute Euclidean Distances (ED) from target (OLD projection)
        old_trg = old_rsc[idx_trg,:]
        
        old_dist = [euclidean(old_fb_pos[i,:],old_trg) 
                    for i in range(old_fb_pos.shape[0])]
  


        ###----------- RUN NEW PROJECTION -----------###   

        print('Testing the NEW projection offline ...')
        
        # pick the subject's json file for the NEW projection (the modern MDS was used for RS generation) 
        sub_json_file = os.path.join(datadir,sub,'new_proj',sub+'.json')
        
        # create an empty RSA object for the NEW projection
        rtRSAObj = nfrsa.rtRSA(' ',2,' ')
        
        # load properties in the rtRSAObj
        rtRSAObj.load(sub_json_file)        
        
        
        # define a variable to store the positions of the FB stimulus in time (for the NEW projection)
        stimulus_positions = np.empty((nr_of_trials,2))

        
        # define the output directory for the NEW projection results
        plot_outdir = os.path.join(datadir,
                                   sub,
                                   'new_proj',
                                   session)
        os.makedirs(plot_outdir,exist_ok=True)
        
        # RS centering 
        print('Moving the origin of the RS to the target coordinates ...')
        rtRSAObj.RS_coords = rtRSAObj.RS_coords - rtRSAObj.RS_coords[idx_trg,:]
        
        scat, ax, line = create_baseline_figure(rtRSAObj, plot_outdir)
        
        print('Run new projection ...')
        for idx_fb, tvalues in enumerate(nf_run_tvalues):
             
            # define the initial point of the optimisation procedure
            if idx_fb == 0:
                theta = 45*math.pi/180 # start the minimisation with an arbitrary angle of 45° 
                
            else:
                #start with the same angular position of the previous feedback point
                theta = math.acos(stimulus_positions[idx_fb-1,0]/np.linalg.norm(stimulus_positions[idx_fb-1],ord=2))
                if stimulus_positions[idx_fb-1,1] < 0:
                    theta = -theta
                    
            t = time.perf_counter()    
            stimulus_positions[idx_fb,0], \
                stimulus_positions[idx_fb,1], flag, x0 ,n_iter, fun_eval, distances = optimise_projection(rtRSAObj, 
                                                                       tvalues,
                                                                       idx_trg, 
                                                                       theta)
                                                        
          
            timer.append(time.perf_counter()-t)
            all_x0.append(x0) 
            num_iter.append(n_iter)
            fun_vals.append(fun_eval)
            all_flag.append(flag)
            
            scat, line = create_feedback(scat, 
                                         ax, 
                                         line, 
                                         idx_fb, 
                                         stimulus_positions, 
                                         rtRSAObj.RS_coords, 
                                         plot_outdir)
         
               
        all_fb_positions.append(stimulus_positions)        

        # compute Euclidean Distances from target (NEW projection)
        new_trg = rtRSAObj.RS_coords[idx_trg,:]
        
        new_dist = [euclidean(stimulus_positions[i,:],new_trg) 
                    for i in range(stimulus_positions.shape[0])]
        
        
        
        all_ED += old_dist + new_dist
        repeated_sub_list += 2*len(old_dist)*[sub]
        repeated_session_list += 2*len(old_dist)*[session.split('_')[2]]
        prj_type += len(old_dist)*['OLD']
        prj_type += len(new_dist)*['NEW']
        
        
        #correlation between target CD and target ED 
        old_corr = pearsonr(fb_distances[:,idx_trg],old_dist)[0]
        old_p_corr = pearsonr(fb_distances[:,idx_trg],old_dist)[1]

        new_corr = pearsonr(fb_distances[:,idx_trg],new_dist)[0]
        new_p_corr = pearsonr(fb_distances[:,idx_trg],new_dist)[1]


    
        #ouptut generation
        output['Subject'] += 2*[sub]
        output['Run'] += 2*[session.split('_')[2]]
        output['ED-CD Correlation'] += [old_corr, new_corr]
        output['ED-CD p-value Correlation'] += [old_p_corr, new_p_corr]
        output['Projection type'] += ['OLD']+['NEW']
        output['FB_position'] += all_fb_positions
        output['Success flag'] += all_flag
        output['Initial_position'] += all_x0
        output['Num iter'] += num_iter
        output['Fun eval'] += fun_vals
        output['Execution time'] += timer
        
        print('Done.')
        
        orig_corr_distances += 2*list(fb_distances[:,idx_trg]) #2 times for plotting
        plt.close('all')

pickle.dump(output, open(os.path.join(datadir,'opt_output.pkl'),'wb'))
