# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:10:21 2023

@author: aciarlo@unisa.it

"""


import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


#packages for the optimization
from scipy.optimize import LinearConstraint, SR1 #tested using scipy version 1.5.2
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import math



#function for the computation of an approximate solution if the optimisation fails   
def approx_segment(distances, RScoords, triangle_flag = False): #SEGMENT
    
    '''
    This function approximates the computation of the position of a new projected point 
    placing it on the segment between the two base stimuli with the lowest CD from the
    current pattern and its closest stimulus.
    
    distances: CDs between current pattern and base stiluli
    RScoords: target-centered coordinates of the base stimuli on the RS
        
    '''
    idx_cd_min = np.argmin(distances)
    
       
    if distances[idx_cd_min] < 1e-3 : #arbitrary
        
        return RScoords[idx_cd_min,0], RScoords[idx_cd_min,1]
    
    else:
        
        #compute ED of all base stimuli from the idx_cd_min stimulus (stim0)
        ed = []
        ed.append([euclidean(RScoords[idx_cd_min,:], RScoords[i,:]) 
                    for i in range(len(RScoords))])
        idx_ed_min = np.argsort(ed)[0,1] # take the second closest stimulus to stim0 (stim1), the first is stim0 itself
        
        #distance between the two selected stimuli
        d = euclidean(RScoords[idx_cd_min], RScoords[idx_ed_min])
        
        #compute distance from  stim0 (d0) from the intersection {d0:d1 = CD0:CD1, d0+d1 = d}
        d0 = d*distances[idx_cd_min]/(distances[idx_cd_min]+distances[idx_ed_min])
            
        coef = np.polyfit(RScoords[[idx_cd_min, idx_ed_min],0], RScoords[[idx_cd_min, idx_ed_min],1], 1)
        
        a = 1+(coef[0])**2
        b=2*(coef[0]*(coef[1]-RScoords[idx_cd_min,1])-RScoords[idx_cd_min,0])
        c= RScoords[idx_cd_min,0]**2+(coef[1]-RScoords[idx_cd_min,1])**2 - d0**2
        delta = (b)**2-4*a*c
        
        # compute x coordinate of the point belonging to the segment 0-1, with distance d0 from stim0
        x_hat1 = (-b+np.sqrt(delta))/(2*a)
        x_hat2 = (-b-np.sqrt(delta))/(2*a)
        
        #check which solution falls between stim0 and stim1
        if x_hat1 >= min(RScoords[[idx_cd_min,idx_ed_min],0]) and x_hat1 <= max(RScoords[[idx_cd_min,idx_ed_min],0]): 
            x_hat = x_hat1
        else:
            x_hat = x_hat2
            
        # comupte y
        y_hat = coef[0]*x_hat + coef[1]
        
        # fig, ax = plt.subplots()
        # ax.scatter(RScoords[:,0],RScoords[:,1],color='r')
        # ax.plot(x_hat,y_hat,'*',color='g')
        # fig.savefig(os.getcwd()+'/check_segment.jpg')       
        
        if triangle_flag:
            return x_hat, y_hat, coef[0], idx_ed_min
        else:
            return x_hat, y_hat



#function for the computation of an approximate solution if the optimization fails   

def approx_triangle(distances, RScoords,idx_trg): #TRIANGLE

    '''
    This function starts from the segment solution and project the new point on 
    a line orthogonal to the segment and so that the distance from the target is exactly CD-target
    
    '''
    
    def check_constraints(idx_cd_min,RScoords, cur_sol):
        
        '''
        check if the current solution respects the constraints
        
        '''
        
        ed = []
        ed.append([euclidean(cur_sol, RScoords[i,:]) 
                    for i in range(len(RScoords))])
        idx_ed_min = np.argmin(ed)

        return idx_ed_min == idx_cd_min
    
    idx_cd_min = np.argmin(distances)
    x_hat, y_hat, m, idx_ed_min = approx_segment(distances, RScoords, triangle_flag = True) # SEGMENT
    
    d_target = distances[idx_trg]
   
    
    # coefficients of 2nr order equation equation
    a = 1+1/m**2
    b=-2/m*(x_hat/m+y_hat)
    c = (x_hat/m+y_hat)**2 - d_target**2
    
    delta = b**2-4*a*c
    # NOTE: delta might be less than 0 in case the circle centered to the target 
    # and the line orthogonal to the segment don't intersect 
    
    #compute the coordinate of the RS centroid G (to project the point 'outside' the RS)
    xg = np.mean(RScoords,axis=0)[0]
    yg = np.mean(RScoords,axis=0)[1]

    # NOTE: the code below could be imporved just checking that the point is outside the poligonal area defined by the base stimuli
    # not implemented now

    if (RScoords[idx_cd_min,0] -xg)*(RScoords[idx_ed_min,0]-xg)>0 and (RScoords[idx_cd_min,1] -yg)*(RScoords[idx_ed_min,1]-yg)<0 and delta>=0: 
        #both left or both right (one above and one below G)
        #plot always outside the space
        if x_hat <= xg:
            x = (-b - math.sqrt(delta))/(2*a) 
        else:
            x = (-b + math.sqrt(delta))/(2*a)
            
        y = -1/m*(x-x_hat)+y_hat
        
        if check_constraints(idx_cd_min,RScoords, [x,y]):
            flag = 'approx_triangle'    
            return x, y, flag
        else:
            flag = 'approx_segment'
            return x_hat, y_hat,flag
        
    elif (RScoords[idx_cd_min,1] -yg)*(RScoords[idx_ed_min,1]-yg)>0 and (RScoords[idx_cd_min,0] -xg)*(RScoords[idx_ed_min,0]-xg)<0 and delta>=0:
        #both above or both below (one at left and one at right of G)
        #plot always outside the space
        if y_hat <= yg:
            x = (-b+np.sign(m)*math.sqrt(delta))/(2*a)

        else:
            x = (-b-np.sign(m)*math.sqrt(delta))/(2*a)
            
        y = -1/m*(x-x_hat)+y_hat
        
        if check_constraints(idx_cd_min,RScoords, [x,y]):
            flag = 'approx_triangle'    
            return x, y, flag
        else:
            flag = 'approx_segment'
            return x_hat, y_hat,flag
    
    elif (RScoords[idx_cd_min,0] -xg)*(RScoords[idx_ed_min,0]-xg)>0 and (RScoords[idx_cd_min,1] -yg)*(RScoords[idx_ed_min,1]-yg)>0 and delta>=0: 
        #both above or below and left or right
        if y_hat <= yg:
            x = (-b+np.sign(m)*math.sqrt(delta))/(2*a)

        else:
            x = (-b-np.sign(m)*math.sqrt(delta))/(2*a)
        
        y = -1/m*(x-x_hat)+y_hat
        
        if check_constraints(idx_cd_min,RScoords, [x,y]):
            flag = 'approx_triangle'    
            return x, y, flag
        else:
            flag = 'approx_segment'
            return x_hat, y_hat,flag
    
    else: 
        #return segment solution
        
        flag = 'approx_segment'
        return x_hat, y_hat,flag

    
    # fig1, ax1 = plt.subplots()
    # ax1.scatter(RScoords[:,0],RScoords[:,1],color='r')
    # ax1.plot(x,y,'*',color='g')
    # fig1.savefig(os.getcwd()+'/check_triangle.jpg')  
    
   
        
    
    
#objective function
def fun(x,d):
    
    '''
    Function to minimise
    x: coordinates to be optimised (target-centered)
    d: distance from the target to be represented
        
    '''
    
    return abs(x[0]**2 + x[1]**2 - d**2)


#jacobian of the objective function
def fun_jac(x,d):
    
    '''
    Jacobian matrix of the objective function
    x: coordinates to be optimised
    d: distance from the target to be represented
    '''
    tmp = x[0]**2 + x[1]**2 - d**2
    
    return np.array([2*x[0], 2*x[1]]) * np.sign(tmp) #tmp / abs(tmp)

#hessian of the objective function
def fun_hes(x,d):
    
    '''
    Hessian matrix of the objective function
    x: coordinates to be optimised
    d: distance from the target to be represented
    '''
    tmp = x[0]**2 + x[1]**2 - d**2
    
    return np.diag([2, 2]) * np.sign(tmp) #tmp / abs(tmp)

    
    
#optimisation procedure
def optimise_projection(rtRSAObj, new_stim, idx_trg,theta):
    
    """

    Parameters
    ----------
    rtRSAObj : rtRSA object
        rtRSA object that represents the RS currently used.
    new_stim : numpy array
        t statistics associated with the current time point. These values,
        usually are extracted from TurboBrainVoyager.
    idx_trg : int
        Index (from 0 to 4) of the target.
    theta : float
        angle (in radiant) to place the initial point of the minimisation problem on the circle

    Returns
    -------
    res.x[0]: numpy float
        x-coordinate of the projected point.
    res.x[1]: numpy float
        y-coordinate of the projected point.
    flag : string
        success flags: 'approx_' = approximate method, 'exact' = exact method


    References:
    [1] https://docs.scipy.org/doc/scipy-1.5.2/reference/tutorial/optimize.html#trust-region-constrained-algorithm-method-trust-constr

    """
 
    #the number of stimuli equals the number of columns 
    NrOfBaseStim = np.shape(rtRSAObj.base_stimuli)[1]

    if rtRSAObj.dist_metric == 1:
        # Use the Pearson correlation and transform it in a distance measure
        distances = [(1-np.corrcoef(rtRSAObj.base_stimuli[:,idx], new_stim)[0][1],idx) 
        for idx in range(NrOfBaseStim)]

    if rtRSAObj.dist_metric == 2:
        # Use the classical euclidean distance
        # not tested here
        return


    if rtRSAObj.dist_metric == 3:
        # Use absolute activation difference
        # not tested here
        return


    unordered_distances = np.array(distances)
    
    if min(unordered_distances[:,0]) <=1e-3: #call approximate solution to ensure convergence
        
        x,y = approx_segment(unordered_distances[:,0],rtRSAObj.RS_coords) #return minimum
       
        
        return x,y,'approx_min',[float('nan'),float('nan')],int('nan'),float('nan'), distances
    
    else:
        
        # print('unordered distances:',unordered_distances)
        #order according to the first column, e.g. CD
        ordered_distances = unordered_distances[unordered_distances[:, 0].argsort()]  
        # print('ordered distances:',ordered_distances)
    
        # matrix with difference between coordinates pairs
        left_constrain_matrix = []
        # matrix with difference between l2-norm of coordinates pairs
        right_constrain_matrix = []
        
        
        # fill the matrices for the optimization
        # loop from the first to the penultimate
        for i in range(ordered_distances.shape[0]-1):
            
            #get the two coordinates
            x1 = rtRSAObj.RS_coords[int(ordered_distances[i,1])] 
            
            for j in range(i+1,ordered_distances.shape[0]):
                x2 = rtRSAObj.RS_coords[int(ordered_distances[j,1])]
            
                left_constrain_matrix.append(x1-x2)
                right_constrain_matrix.append(0.5*(np.linalg.norm(x1,2)**2 - np.linalg.norm(x2,2)**2))
            
    
        left_constrain_matrix = -1*np.array(left_constrain_matrix)
        right_constrain_matrix = -1*np.array(right_constrain_matrix)
        lb = -np.inf*np.ones((left_constrain_matrix.shape[0],1))
        lb = lb.ravel()
        
        #linear constraints in scipy "format"
        linear_constraint = LinearConstraint(left_constrain_matrix,
                                             lb,
                                             right_constrain_matrix)
        
       
        # starting point on the circle with the target in the center at an angle theta.
        x0 =[unordered_distances[idx_trg,0]*math.cos(theta),
             unordered_distances[idx_trg,0]*math.sin(theta)] 
    
        
        # optimisation with scipy
        res = minimize(fun,
                       x0, 
                       args = (unordered_distances[idx_trg,0]), 
                       constraints = linear_constraint, 
                       method='trust-constr',
                       jac = fun_jac, 
                       hess = fun_hes,
                       options={'maxiter':400,'disp' : False, 'barrier_tol' : 1e-8,
                                'initial_constr_penalty':10}
                        )
        
        
        if not(res.success) or res.fun> 0.1 or res.constr_violation > 0:
            
          
            x,y, flag = approx_triangle(unordered_distances[:,0],rtRSAObj.RS_coords,idx_trg) # TRIANGLE 
            
            return x,y,flag,[float('nan'),float('nan')],res.nit,res.fun, distances
             
        else:
            
            return res.x[0],res.x[1],'exact',x0,res.nit,res.fun, distances
   
    


def create_baseline_figure(rtRSAObj,outdir):
    '''
    
    Parameters
    ----------
    rtRSAObj : Object class rtRSA
        rtRSA object that contains all the info for the experiments (e.g. 
        coordinates of the RS space, the inversion matrix and the base stimuli).
    outdir : string
        Output directory to save the feedback images

    Returns
    -------
    scat : mtplotlib scatter plot 
        Instance of the created scatter plot that has to be updated.
    ax : matplotlib axis
        Axis of the scatter plot.
    line : matplotlib line
        Path of the previous feedback locations

    '''
    

    
    fig,ax = plt.subplots()
    fig.set_facecolor('dimgray')
    scat = ax.scatter(rtRSAObj.RS_coords[:,0],rtRSAObj.RS_coords[:,1],s=150, c='yellow',
               edgecolors = 'black', zorder = 2)
    
    line = ax.plot(rtRSAObj.RS_coords[0:1,0],rtRSAObj.RS_coords[0:1,1], color = 'dimgray', zorder = 1)
    
    ax.axis('off')
    ax.axis('equal')

    for label, x, y in zip(rtRSAObj.conditions, 
                           rtRSAObj.RS_coords[:,0],
                           rtRSAObj.RS_coords[:,1]):
        ax.annotate(label, xy=(x, y), xytext=(-20, 20),size=15,
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), zorder=2) 
    
    
    yellow_circle = mlines.Line2D([], [], color='yellow',  linestyle='None', 
                                  marker='o',markeredgecolor='black', markeredgewidth=0.5,
                              markersize=15, label='Base stimuli')
    red_star = mlines.Line2D([], [], color='red',linestyle='None',  marker='*',
                             markeredgecolor='black', markeredgewidth=0.5,
                              markersize=15, label='Current stimulus')
    
    ax.legend(handles=[yellow_circle, red_star],loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4)
    
    
    plt.savefig(os.path.join(outdir,'initial_img.png'), dpi=200,
                facecolor='dimgray', bbox_inches='tight') 
    
    #plt.close('all')
        
    return scat, ax, line


def create_feedback(scat,ax, line, idx_fb,stimulus_positions,RScoords, outdir):
    '''
    

    Parameters
    ----------
    scat : scatter plot instance
        The initial scatter plot that has to be updated.
    ax : matplotlib axis
        The axes of the initial scatter plot.
    line : matplotlib line
        Path of the previous feedback locations.
    idx_fb : int
        Index to access the stimulus positions array.
    stimulus_positions : numpy array
        Array that contains the stimulus positions for each feedback block.
    RScoords : numpy array 
        Array of the RS base coordinates
    outdir : string
        Output directory to save the feedback images

    Returns
    -------
    new_scat : matplotlib scatter plot 
        The matplot instance of the new scatter plot. It will be used to delete
        in the next iteration the red star that indicates the previous feedback
    line : matplotlib line
        Path of the previous feedback locations

    '''
 

    if idx_fb == 0:
        #first feedback        
        new_scat = ax.scatter(stimulus_positions[idx_fb,0],stimulus_positions[idx_fb,1], 
                    marker = '*',s=200, color = 'red', edgecolors='black', zorder = 3)
    
    elif idx_fb > 0 and idx_fb < 4:
        #from the second feedback to the fourth
        new_scat = scat
        #delete the reference to the point plotted for the previous iteration
        new_scat.set_offsets(np.delete(scat.get_offsets(), 0, axis=0))

        line.pop(0).remove()
        #plot the new position of the current mental state and, in dashed lines,
        #the trajectory until now        
        new_scat = ax.scatter(stimulus_positions[idx_fb,0],stimulus_positions[idx_fb,1], 
                    marker = '*',s=200, color = 'red', edgecolors='black',zorder=3)
        line = ax.plot(stimulus_positions[:idx_fb+1,0],stimulus_positions[:idx_fb+1,1], '--',
                        color = 'black',alpha=0.2,zorder=1) 

    else:
        #from the fourth feedback onwards
        new_scat = scat
        #delete the reference to the point plotted for the previous iteration
        new_scat.set_offsets(np.delete(scat.get_offsets(), 0, axis=0))
        
        line.pop(0).remove()
        #plot the new position of the current mental state and, in dashed lines,
        #the trajectory until now        
        new_scat = ax.scatter(stimulus_positions[idx_fb,0],stimulus_positions[idx_fb,1], 
                    marker = '*',s=200, color = 'red', edgecolors='black',zorder=3)
        line = ax.plot(stimulus_positions[(idx_fb-4):(idx_fb+1),0],stimulus_positions[(idx_fb-4):(idx_fb+1),1], '--',
                        color = 'black',alpha=0.2,zorder=1)

    ax.set_facecolor('dimgray')

	#scale plot
    
    xmin = np.min(np.append(RScoords[:,0], stimulus_positions[idx_fb,0]))
    xmax = np.max(np.append(RScoords[:,0], stimulus_positions[idx_fb,0]))
    ymin = np.min(np.append(RScoords[:,1], stimulus_positions[idx_fb,1]))
    ymax = np.max(np.append(RScoords[:,1], stimulus_positions[idx_fb,1]))
    dx = (xmax-xmin)*0.1
    dy = (ymax - ymin)*0.1
    ax.set_xlim(xmin - dx , xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    

    #plt.xticks([])
    #plt.yticks([])
    #plt.axis('off')
            
    #save figure
    plt.savefig(os.path.join(outdir,'tvals_Trial' + str(idx_fb)+ '.png'),
               facecolor='dimgray', edgecolor='none', dpi=200, bbox_inches='tight')
    #plt.close('all')

    
    return new_scat, line