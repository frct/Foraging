# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:51:15 2021

@author: franc
"""

# prelude

import RateComparisonModel_TA as model
import matplotlib.pyplot as plt
import numpy as np
import import_pycontrol as ip
from scipy import optimize

def InitialisationIndices(n_parameters):
    nx = 3**n_parameters

    I = np.zeros((nx, n_parameters))

    for idy in range(n_parameters): # sweeping through columns
        idx = 0
        for n_reps in range(3 ** idy): # each column is made of repeated sequences of groups of 1's 2's 3's. The first colums containts only one repetition, the next 3, the next 9, etc...
            for a in range(3):
                for b in range(3**(n_parameters - idy-1)): # size of this repetition
                    I[idx, idy] = a
                    idx = idx + 1
                    
    return I.astype(int)

# set variables of the model

dt = 100 # time step size in ms

alpha_env = [0.005, 0.01, 0.02]
alpha_patch = [0.2, 0.5, 0.8]
beta = [1, 2, 5]
offset = [0.05, 0.1, 0.2]
bias = [0.5, 1, 2]
bnds = ((0,0.1), (0,1), (0.001, 100), (0,1), (0,100))

initialisation_points = np.array([alpha_env, alpha_patch, beta, offset, bias])
n_parameters = initialisation_points.shape[0]
n_initialisations = 3**n_parameters
combination_of_initialisations = InitialisationIndices(n_parameters)

# load data

# change to location of raw data (.txt files)
behaviour_data_folder = '../../raw_data/behaviour_data/'

experiment = ip.Experiment(behaviour_data_folder)

mouse1_data = experiment.get_sessions(1)

n_sessions = len(mouse1_data)

session_parameters = []

for session in range(n_sessions):
    this_session = mouse1_data[session]
    print("session %s of %s" %(session+1, n_sessions))
#this_session = ip.Session('../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt')
# this_session = ip.Session('../../raw_data/behaviour_data/fp10-2019-05-20-132138.txt')
    model.store_variables(this_session, dt)
# test = model.FocusedPatchTransitions(this_session)
# r = model.FocusedRewardTimes(this_session)


    optimised_parameters = np.zeros((n_initialisations, n_parameters))
    test = np.zeros((n_initialisations))
    negLL = np.zeros((n_initialisations))
    avg_likelihood = np.zeros((n_initialisations))
            
    for i in range(n_initialisations):
        #print("initialisation %s of %s " %(i,n_initialisations))
        x0 = np.zeros((n_parameters))
        for p in range(n_parameters):
            x0[p] = initialisation_points[p, combination_of_initialisations[i, p]]
        res = optimize.minimize(
            fun = lambda parameters, this_session : model.LogLikelihood(parameters, this_session),
            args=(this_session),
            x0=x0,
            bounds = bnds)
        optimised_parameters[i,:] = res.x
        test[i] = res.success
        negLL[i] = res.fun
        
    session_parameters.append(optimised_parameters)

#parameters optimised on fp01-2019-02-21-112604
# alpha_env = 0.0001
# alpha_patch = 0.0003
# beta = 4.52
# offset = 1

#parameters optimised on fp10-2019-05-20-132138
alpha_env = 0.0006
alpha_patch = 0.105
beta = 57
offset = 0.35
bias = 6.03

parameters = np.array([alpha_env, alpha_patch, beta, offset, bias])

[rho_env, delta_env, rho_patch, delta_patch, trial_likelihood] = model.RewardRates(parameters,  **this_session.model_vars)

# choices = model.ExtractChoices(this_session)
# [rho_env, delta_env, rho_patch, delta_patch, p_action] = model.RewardRates(this_session, alpha_patch, alpha_env, offset, beta, dt)

# initiate rates

# Choices = model.ExtractChoices(this_session)
# ll = model.LogLikelihood(this_session, alpha_env, alpha_patch, offset, beta, dt)
# [env_rho, env_delta] = model.EnvironmentRewardRate(this_session, alpha_env, time_bin = dt)
# [patch_rho, patch_delta] = model.PatchRewardRate(this_session, alpha_patch, offset=offset, time_bin=dt)

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


# plt.figure()
# plt.plot(np.log(trial_likelihood))

# same with default values

# [rho_env, delta_env, rho_patch, delta_patch, p_action] = model.RewardRates(this_session, time_bin = dt)

# plt.figure()
# plt.plot(rho_env[0:500])
# plt.plot(rho_patch[0:500])


# plt.figure()
# plt.plot(p_action[0:500])

# ll_example = np.sum(np.log(p_action))