# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:03:14 2021

@author: franc
"""

import numpy as np
import BehaviourModels as mdl
from scipy import optimize


def CompressTime(patches, travel_engagement):
    ''' calculates the time stamps of rewards, patch arrivals and departures, 
    and travel block changes when only forage nosepoke time is used.
    If travel_engagement, only use nosepoke time during travel, if not use 
    complete travel time. '''
    
    #initialisation of outputs
    
    total_duration = 0
    
    n_patches = len(patches)
    patch_departures = np.zeros(n_patches)
    patch_arrivals = np.zeros(n_patches)
    reward_times = []
    
    # initialise counters
    
    last_departure = 0
    last_arrival = 0
    
    first_block = patches[0]['block']
    current_block = first_block
    block_transitions = []
    
    for patch_id, patch in enumerate(patches):
        
        if patch['block'] != current_block:
            current_block = patch['block']
            block_transitions.append(last_arrival)                
        
        if travel_engagement: # only consider time in nosepoke holes
            travel_time = round(sum([bout[1] - bout[0] for bout in patch['travel bouts']]),3)
        else:
            if patch != patch[-1]:
                travel_time = round(patch['complete travel time'][1] - patch['complete travel time'][0], 3)
            else:
                travel_time = 0
                
        foraging_time =  sum([sum([bout[1] - bout[0] for bout in reward]) for reward in patch['forage times']]) + sum([bout[1] - bout[0] for bout in patch['forage before travel']])
        total_duration += travel_time + foraging_time
        
        reward_time = last_arrival
        for reward in patch['forage times']:
            reward_time += sum([bout[1] - bout[0] for bout in reward])
            reward_times.append(round(reward_time, 3))
                
        last_departure = round(last_arrival + foraging_time, 3)
        last_arrival = round(last_departure + travel_time, 3)
        patch_departures[patch_id] = last_departure
        
        if patch != patches[-1]:
            patch_arrivals[patch_id+1] = last_arrival
        
    return {"total duration": round(total_duration, 3), "reward times": reward_times, "patch departures": patch_departures, "patch arrivals": patch_arrivals, "block transitions": block_transitions, 'first block': first_block}


def PrepareForOptimisation(session, time_bin=0.1, Engaged = True):
    '''Compute the variables needed for the RL model and store them
    on the session.'''
    
    time_stamps = CompressTime(session, Engaged)
    n_rewards = len(time_stamps['reward times'])
    
    n_bins = int(time_stamps["total duration"] / time_bin) + 1
    model_vars = {
        'reward_times'        : time_stamps["reward times"],
        'departure_times'     : time_stamps["patch departures"],
        'arrival_times'       : time_stamps["patch arrivals"],
        'block_transitions'   : time_stamps["block transitions"],
        'first_block'         : time_stamps['first block'],
        'n_bins'              : n_bins,
        'time_bin'            : time_bin,
        'n_rewards'           : n_rewards,
        'average_reward_rate' : n_rewards / n_bins}
    
    return model_vars


def SessionLogLikelihood(params, model, model_vars): #  
    '''Calculate the LogLikelihood for given parameters and session.  The 
    session must first have been passed to the store_variables function 
    to pre-calculate variables and store them on the session.'''
    if model == 'Reward rate comparison':
        (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving) = mdl.CompRwdRates_mdl(params, **model_vars)      
    elif model == 'no bias':
        params = np.append(params, 0)
        (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving) = mdl.CompRwdRates_mdl(params, **model_vars)
    elif model == 'Constant environment':
        (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving) = mdl.SemiConstantEnv_mdl(params, **model_vars)      
    elif model == 'Constant threshold':
        (threshold, rho_patch, delta_patch, p_action, p_leaving) = mdl.FixedThreshold_mdl(params, **model_vars)
    elif model == 'Single threshold':
        (rho_patch, delta_patch, p_action, p_leaving) = mdl.SingleThreshold_mdl(params, **model_vars)
    elif model == 'Double bias':
        (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving) = mdl.DoubleBias_mdl(params, **model_vars)      
    else:
        raise Exception('invalid model')
    
    loglikelihood = -1 * np.sum(np.log(p_action))
        
    return loglikelihood


def MouseLogLikelihood(sessions, params, model):
    ''' Calculates the log-likelihood of a chosen model summed between all
    sessions of an individual mouse '''
    
    total_log_likelihood = 0
    
    for session in sessions:
        session_log_likelihood = SessionLogLikelihood(params, model, session)
        total_log_likelihood += session_log_likelihood
    
    return total_log_likelihood

def InitialiseOptimisation(model):
    
    if model == 'Reward rate comparison':

        alpha_env = [0, 0.1]
        alpha_patch = [0, 0.2]
        beta = [2, 10]
        reset = [0, 0.5]
        bias = [2, 10]
        n_parameters = 5
        init_range = [alpha_env, alpha_patch, beta, reset, bias]
        bounds = ((0,1), (0,1), (0., 100), (0,1), (0,100))
    
    elif model == 'no bias':
        
        alpha_env = [0, 0.1]
        alpha_patch = [0, 0.2]
        beta = [2, 10]
        reset = [0, 0.5]
        n_parameters = 4
        init_range = [alpha_env, alpha_patch, beta, reset]
        bounds = ((0,1), (0,1), (0., 100), (0,1))
        
    elif model == 'Constant environment':
        
        alpha_env = [0, 0.1]
        alpha_patch = [0, 0.2]
        beta = [2, 10]
        reset = [0, 0.5]
        bias = [2, 10]
        n_parameters = 5
        init_range = [alpha_env, alpha_patch, beta, reset, bias]
        bounds = ((0,1), (0,1), (0., 100), (0,1), (0,100))
        
    elif model == 'Constant threshold':
        
        short_th = [0, 0.5]
        long_th = [0, 0.5]
        alpha = [0, 0.2]
        beta = [2, 10]
        reset = [0, 0.5]
        bias = [2, 10]
        n_parameters = 6
        init_range = [short_th, long_th, alpha, beta, reset, bias]
        bounds = ((0,1), (0,1), (0,1), (0., 100), (0,1), (0,100))
        
    elif model == 'Single threshold':
        
        threshold = [0, 0.5]
        alpha = [0, 0.2]
        beta = [2, 10]
        reset = [0, 0.5]
        bias = [2, 10]
        n_parameters = 5
        init_range = [threshold, alpha, beta, reset, bias]
        bounds = ((0,1), (0,1), (0., 100), (0,1), (0,100))
        
    elif model == 'Double bias':
        alpha_env = [0, 0.1]
        alpha_patch = [0, 0.2]
        beta = [2, 10]
        reset = [0, 0.5]
        bias_long = [2, 10]
        bias_short = [2, 10]
        n_parameters = 6
        init_range = [alpha_env, alpha_patch, beta, reset, bias_long, bias_short]
        bounds = ((0,1), (0,1), (0., 100), (0,1), (0,100), (0,100))
        
    return n_parameters, init_range, bounds

def TestConvergence(optimisation_results, negLL):
    
    best_idx = np.argmin(negLL)
    best_params = optimisation_results[best_idx]
    
    optimisation_results = np.array(optimisation_results)
    n_initialisations, n_parameters = optimisation_results.shape

    converged_idx = []
    
    for i in range(n_initialisations):
        converged = True
        for j in range(n_parameters):
            param = optimisation_results[i,j]
            if param > best_params[j] * 1.05 or param < best_params[j] * 0.95:
                converged = False
        if converged:
            converged_idx.append(i)
    
    if len(converged_idx) >= 3:
        test = True
    else:
        test = False
    
    return test


def Optimise(sessions, model):
    
    n_parameters, init_range, bnds = InitialiseOptimisation(model)
    
    min_initialisations = 10
    max_initialisations = 20
    
    optimised_parameters = []
    negLL = []
    
    converged = False
    n_init = 0
    
    initial_points = []
    
    while (n_init < min_initialisations or not converged) and n_init < max_initialisations:
        
        print('initialisation ' + str(n_init + 1))
        
        x0 = np.zeros((n_parameters))
        for p in range(n_parameters):
            x0[p] = np.random.uniform(init_range[p][0], init_range[p][1])
            
        initial_points.append(x0)
        res = optimize.minimize(
            fun = lambda parameters, model, sessions : MouseLogLikelihood(sessions, parameters, model),
            args=(model,sessions),
            x0=x0,
            bounds = bnds)
        
        if res.success: # only save this optimisation if it successfully converged
            optimised_parameters.append(res.x)
            negLL.append(res.fun)
            n_init += 1
            
        
        if n_init >= min_initialisations:
            converged = TestConvergence(optimised_parameters, negLL)

    best_idx = np.argmin(negLL)
    best_parameters = optimised_parameters[best_idx]
    
    results = {'initial points': initial_points, 'optimisation results': optimised_parameters, 'minima': negLL, 'best result': best_parameters}
        
    return results