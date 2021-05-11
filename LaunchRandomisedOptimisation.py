# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:17:48 2021

@author: franc
"""

import RateComparisonModel as model
import matplotlib.pyplot as plt
import numpy as np
import import_pycontrol as ip
from scipy import optimize

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

behaviour_data_folder = '../../raw_data/behaviour_data/'

experiment = ip.Experiment(behaviour_data_folder)

mice = experiment.subject_IDs

optimisation_results = []

for mouse in mice:
    print("mouse number %s" %(mouse))
    mouse_data = experiment.get_sessions(mouse)
    
    n_sessions = len(mouse_data)
    
    mouse_results = {"n converged": np.zeros((n_sessions)), "convergence": np.zeros((n_sessions)), "negative log-likelihood": np.zeros((n_sessions)), "parameters": np.zeros((n_sessions, n_parameters))}
    
    for session_idx, this_session in enumerate(mouse_data):
        
        print("session number %s of %s" %(session_idx+1, n_sessions))
        #this_session = ip.Session('../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt')
        model.store_variables(this_session, dt, False)
        
        
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
                fun = lambda parameters, this_session : model.LogLikelihood(parameters, this_session),
                args=(this_session),
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
    
    optimisation_results.append(mouse_results)
        
# best_parameters = np.array([0.2635, 0.2617, 100, 0, 3.165])

[rho_env, delta_env, rho_patch, delta_patch, trial_likelihood] = model.RewardRates(best_parameters,  **this_session.model_vars)

plt.figure()
plt.plot(rho_env)
plt.plot(rho_patch)

plt.figure()
plt.plot(rho_env)

plt.figure()
plt.plot(trial_likelihood)

plt.figure()
plt.plot(rho_env[0:500])
plt.plot(rho_patch[0:500])

plt.figure()
plt.plot(trial_likelihood[0:500])
