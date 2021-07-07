# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:17:48 2021

@author: franc
"""

import RateComparisonModel as model
import numpy as np
from scipy import optimize
import os
import SingleSessionAnalysis as ssa

# set variables of the model

dt = 100 # time step size in ms

alpha_env = [0, 0.1]
alpha_patch = [0, 0.2]
beta = [2, 10]
reset = [0, 0.5]
bias = [2, 10]
n_parameters = 5
initialisation_bounds = [alpha_env, alpha_patch, beta, reset, bias]

bnds = ((0,1), (0,1), (0.001, 100), (0,1), (0,100))

n_initialisations = 100

mice = [1,2,3,7,8,10,11,12,13,14,16,17]

optimisation_results = {}

for mouse in mice:
    print("mouse number %s" %(mouse))
    
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    
    files = os.listdir(folder_path)
    n_sessions = len(files)
    
    mouse_tag = 'mouse ' + str(mouse)
    mouse_results = {"n converged": np.zeros((n_sessions)), "convergence": np.zeros((n_sessions)), "negative log-likelihood": np.zeros((n_sessions)), "parameters": np.zeros((n_sessions, n_parameters))}
    
    for session_idx, file in enumerate(files):
        
        print("session number %s of %s" %(session_idx+1, n_sessions))
        filepath = folder_path + '/' + file
        
        this_session, subject, date = ssa.ExtractPatches(filepath)

        model_vars = model.PrepareForOptimisation(this_session, dt, False)
    
        initialisation_points = np.zeros((n_initialisations, n_parameters))
        optimised_parameters = np.zeros((n_initialisations, n_parameters))
        test = np.zeros((n_initialisations))
        results = np.zeros((n_initialisations))
        avg_likelihood = np.zeros((n_initialisations))
              
        for i in range(n_initialisations):
            #print("initialisation %s of %s " %(i,n_initialisations))
            x0 = np.zeros((n_parameters))
            for p in range(n_parameters):
                x0[p] = np.random.uniform(initialisation_bounds[p][0], initialisation_bounds[p][1])
            res = optimize.minimize(
                fun = lambda parameters, model_vars : model.LogLikelihood(parameters, model_vars),
                args=(model_vars),
                x0=x0,
                bounds = bnds)
            optimised_parameters[i,:] = res.x
            test[i] = res.success
            results[i] = res.fun
        
        # only keep results which have converged
        negLL = [x for i, x in enumerate(results) if test[i]] 
        converged_optimisations = [optimised_parameters[i,:] for i in range(n_initialisations) if test[i]]
        
        best_idx = np.argmin(negLL)
        list_of_all_best_idx = [i for i, x in enumerate(negLL) if 1.01 * min(negLL) > x ]
        convergence_rate = len(list_of_all_best_idx) / len(negLL)
        best_parameters = converged_optimisations[best_idx]
        
        mouse_results["n converged"][session_idx] = len(negLL)
        mouse_results["negative log-likelihood"][session_idx] = negLL[best_idx]
        mouse_results["convergence"][session_idx] = convergence_rate
        mouse_results["parameters"][session_idx, :] = best_parameters
    
    optimisation_results[mouse_tag] = mouse_results