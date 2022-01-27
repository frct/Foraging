# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:15:19 2021

@author: fcinotti
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def FixedThreshold_mdl(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, block_transitions, first_block, n_bins, 
                average_reward_rate):
    '''Changes made to make fuction compatible with numba.jit:
    - All arguments and returned values are now either numpy arrays or
      individual numbers.
    - p_action array is now pre-allocated rather than appended inside loop.
    '''
    # unpack model parameters.
    short_threshold, long_threshold, alpha, beta, reset, bias = params

    
    # initialise variables
    threshold = np.zeros(n_bins+1)
    if first_block == 'short':
        threshold[0] = short_threshold
    elif first_block == 'long':
        threshold[0] = long_threshold
    else:
        raise Exception('Unknown first block')
    
    rho_patch = np.zeros(n_bins+1)
    rho_patch[0] = reset
    delta_patch = np.zeros(n_bins+1)
    
    next_reward_index = 0
    first_bin = True
    #nd = 0 # Counter for number of decison points.
    
    it_departures = iter(departure_times)
    arrival_times = np.append(arrival_times[1:], np.inf) # because numba forbids next from having its second default argument, append inf here
    it_arrivals = iter(arrival_times)
    block_transitions.append(np.inf) # because numba forbids next from having its second default argument, append inf here
    it_transitions = iter(block_transitions)
    
    departure = next(it_departures)
    arrival = next(it_arrivals)
    transition = next(it_transitions)
    
    p_action = np.zeros(n_bins)
    p_leaving = np.zeros(n_bins)
       
    for i in range(1, n_bins+1):
        t = round(i * time_bin, 1)
        
        #check current block type and select corresponding threshold
        if t > transition:
            transition = next(it_transitions)
            if threshold[i-1] == short_threshold:
                threshold[i] = long_threshold
            elif threshold[i-1] == long_threshold:
                threshold[i] = short_threshold
        
        # check if the upcoming reward is in this time bin, and if so start looking out for the next reward
        
        if next_reward_index < n_rewards:
            r = 0 if reward_times[next_reward_index] > t else 1        
            if r == 1:
                next_reward_index += 1
        else:
            r = 0

        
        #if the animal has stayed in the patch throughout this time bin, update the patch estimates and compute probability of staying after update
        if t < departure:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha * delta_patch[i]
            p_action[i] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * threshold[i-1]))            
            p_leaving[i] = 1 - p_action[i]
            # p_action[nd] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * threshold[i-1]))            
            # p_leaving[nd] = 1 - p_action[nd]
            # nd += 1
        
        # if travelling keep rho_patch constant
        elif t >= departure:
            
            if first_bin: #update rho_patch one last time and compute probability of leaving
                delta_patch[i] = r - rho_patch[i-1]
                rho_patch[i] = rho_patch[i-1] + alpha * delta_patch[i]
                p_action[i] = np.exp(beta * threshold[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * threshold[i-1]))
                p_leaving[i] = p_action[i]                
                # p_action[nd] = np.exp(beta * threshold[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * threshold[i-1]))
                # p_leaving[nd] = p_action[nd]
                # nd += 1
                first_bin = False
            else:
                delta_patch[i] = 0
                rho_patch[i] = reset
                p_action[i] = 1
                p_leaving[i] = 0
                
            if t > arrival:
                
                departure = next(it_departures)
                arrival = next(it_arrivals)
                first_bin = True

    # p_action = p_action[:nd]
    # p_leaving = p_leaving[:nd]
    return (threshold, rho_patch, delta_patch, p_action, p_leaving)

@jit(nopython=True)
def SingleThreshold_mdl(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, block_transitions, first_block, n_bins, 
                average_reward_rate):
    '''Changes made to make fuction compatible with numba.jit:
    - All arguments and returned values are now either numpy arrays or
      individual numbers.
    - p_action array is now pre-allocated rather than appended inside loop.
    '''
    # unpack model parameters.
    threshold, alpha, beta, reset, bias = params

    
    # initialise variables
    
    rho_patch = np.zeros(n_bins+1)
    rho_patch[0] = reset
    delta_patch = np.zeros(n_bins+1)
    
    next_reward_index = 0
    first_bin = True
    nd = 0 # Counter for number of decison points.
    
    it_departures = iter(departure_times)
    arrival_times = np.append(arrival_times[1:], np.inf) # because numba forbids next from having its second default argument, append inf here
    it_arrivals = iter(arrival_times)
    
    departure = next(it_departures)
    arrival = next(it_arrivals)
    
    p_action = np.zeros(n_bins)
    p_leaving = np.zeros(n_bins)
       
    for i in range(1, n_bins+1):
        t = round(i * time_bin, 1)

        
        # check if the upcoming reward is in this time bin, and if so start looking out for the next reward
        
        if next_reward_index < n_rewards:
            r = 0 if reward_times[next_reward_index] > t else 1        
            if r == 1:
                next_reward_index += 1
        else:
            r = 0

        
        #if the animal has stayed in the patch throughout this time bin, update the patch estimates and compute probability of staying after update
        if t < departure:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha * delta_patch[i]
            p_action[nd] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * threshold))            
            p_leaving[nd] = 1 - p_action[nd]
            nd += 1
        
        # if travelling keep rho_patch constant
        elif t >= departure:
            
            if first_bin: #update rho_patch one last time and compute probability of leaving
                delta_patch[i] = r - rho_patch[i-1]
                rho_patch[i] = rho_patch[i-1] + alpha * delta_patch[i]
                p_action[nd] = np.exp(beta * threshold) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * threshold))
                p_leaving[nd] = p_action[nd]
                nd += 1
                first_bin = False
            else:
                delta_patch[i] = 0
                rho_patch[i] = reset
                
            if t > arrival:
                
                departure = next(it_departures)
                arrival = next(it_arrivals)
                first_bin = True

    p_action = p_action[:nd]
    p_leaving = p_leaving[:nd]
    return (rho_patch, delta_patch, p_action, p_leaving)

@jit(nopython=True)
def CompRwdRates_mdl(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, block_transitions, first_block, n_bins, 
                average_reward_rate):
    '''Changes made to make fuction compatible with numba.jit:
    - All arguments and returned values are now either numpy arrays or
      individual numbers.
    - p_action array is now pre-allocated rather than appended inside loop.
    '''
    # unpack model parameters.
    alpha_env, alpha_patch, beta, reset, bias = params

    rho_env = np.zeros(n_bins+1)
    rho_env[0] = average_reward_rate
    delta_env = np.zeros(n_bins+1)
    
    rho_patch = np.zeros(n_bins+1)
    rho_patch[0] = reset
    delta_patch = np.zeros(n_bins+1)
    
    next_reward_index = 0
    first_bin = True
    
    it_departures = iter(departure_times)
    arrival_times = np.append(arrival_times[1:], np.inf) # because numba forbids next from having its second default argument, append inf here
    it_arrivals = iter(arrival_times)
    
    departure = next(it_departures)
    arrival = next(it_arrivals)
    
    p_action = np.zeros(n_bins)
    p_leaving = np.zeros(n_bins)
    
    for i in range(1, n_bins+1):
        t = round(i * time_bin, 1)
        
        # check if the upcoming reward is in this time bin, and if so start looking out for the next reward
        
        if next_reward_index < n_rewards:
            r = 0 if reward_times[next_reward_index] > t else 1
            if r == 1:
                next_reward_index += 1
        else:
            r = 0
                
        # update environment reward rate
        delta_env[i] = r-rho_env[i-1]
        rho_env[i] = rho_env[i-1] + alpha_env * delta_env[i] # if r = 1 use alpha_pos, if equal to 0 use alpha_neg  
        
        #if the animal has stayed in the patch throughout this time bin, update the patch estimates and compute probability of staying after update
        if t < departure:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
            p_action[i-1] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))            
            p_leaving[i-1] = 1 - p_action[i-1]
        
        # if travelling keep rho_patch constant
        elif t >= departure:
            
            if first_bin: #update rho_patch one last time and compute probability of leaving
                delta_patch[i] = r - rho_patch[i-1]
                rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
                p_action[i-1] = np.exp(beta * rho_env[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))
                p_leaving[i-1] = p_action[i-1]
                first_bin = False
            else:
                delta_patch[i] = 0
                rho_patch[i] = reset
                p_action[i-1] = 1
                p_leaving[i-1] = 0
                
            if t > arrival:
                
                departure = next(it_departures)
                arrival = next(it_arrivals)
                first_bin = True

    return (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving)

@jit(nopython=True)
def SemiConstantEnv_mdl(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, block_transitions, first_block, n_bins, 
                average_reward_rate):

    # unpack model parameters.
    alpha_env, alpha_patch, beta, reset, bias = params

    rho_env = np.zeros(n_bins+1)
    rho_env[0] = average_reward_rate
    transient_rho_env = np.zeros(n_bins + 1)
    transient_rho_env[0] = average_reward_rate
    delta_env = np.zeros(n_bins+1)
    
    rho_patch = np.zeros(n_bins+1)
    rho_patch[0] = reset
    delta_patch = np.zeros(n_bins+1)
    
    next_reward_index = 0
    #patch_idx = 0
    first_bin = True
    nd = 0 # Counter for number of decison points.
    
    it_departures = iter(departure_times)
    arrival_times = np.append(arrival_times[1:], np.inf) # because numba forbids next from having its second default argument, append inf here
    it_arrivals = iter(arrival_times)
    
    departure = next(it_departures)
    arrival = next(it_arrivals)
    
    p_action = np.zeros(n_bins)
    p_leaving = np.zeros(n_bins)
    
    for i in range(1, n_bins+1):
        t = round(i * time_bin,1)
        #print(t)
        
        # check if the upcoming reward is in this time bin, and if so start looking out for the next reward
        
        if next_reward_index < n_rewards:
            r = 0 if reward_times[next_reward_index] > t else 1
        
            if r == 1:
                next_reward_index += 1
        else:
            r = 0
                
        # update transient environment reward rate, keep threshold constant
        delta_env[i] = r-transient_rho_env[i-1]
        transient_rho_env[i] = transient_rho_env[i-1] + alpha_env * delta_env[i] # if r = 1 use alpha_pos, if equal to 0 use alpha_neg  
        rho_env[i] = rho_env[i-1]
        
        #if the animal has stayed in the patch throughout this time bin, update the patch estimates and compute probability of staying after update
        if t < departure:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
            p_action[nd] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))            
            p_leaving[nd] = 1 - p_action[nd]
            nd += 1
        
        # if travelling keep rho_patch constant
        elif t >= departure:
            
            if first_bin: #update rho_patch one last time and compute probability of leaving
                delta_patch[i] = r - rho_patch[i-1]
                rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
                p_action[nd] = np.exp(beta * rho_env[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))
                p_leaving[nd] = p_action[nd]
                nd += 1
                first_bin = False
            else:
                delta_patch[i] = 0
                rho_patch[i] = reset
                
            if t > arrival: # get next events and reset environment reward rate
                
                departure = next(it_departures)
                arrival = next(it_arrivals)
                #patch_idx += 1
                first_bin = True
                rho_env[i] = transient_rho_env[i]

    p_action = p_action[:nd]
    p_leaving = p_leaving[:nd]
    return (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving)

@jit(nopython=False)
def DoubleBias_mdl(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, block_transitions, first_block, n_bins, 
                average_reward_rate):
    '''Changes made to make fuction compatible with numba.jit:
    - All arguments and returned values are now either numpy arrays or
      individual numbers.
    - p_action array is now pre-allocated rather than appended inside loop.
    '''
    # unpack model parameters.
    alpha_env, alpha_patch, beta, reset, bias_long, bias_short = params

    
    # choose appropriate bias for beginning of the session

    if first_block == 'short':
        bias = bias_short
    elif first_block == 'long':
        bias = bias_long
    else:
        raise Exception('Unknown first block')
    
    rho_env = np.zeros(n_bins+1)
    rho_env[0] = average_reward_rate
    delta_env = np.zeros(n_bins+1)
    
    rho_patch = np.zeros(n_bins+1)
    rho_patch[0] = reset
    delta_patch = np.zeros(n_bins+1)
    
    next_reward_index = 0
    first_bin = True
    
    it_departures = iter(departure_times)
    arrival_times = np.append(arrival_times[1:], np.inf) # because numba forbids next from having its second default argument, append inf here
    it_arrivals = iter(arrival_times)
    block_transitions.append(np.inf) # because numba forbids next from having its second default argument, append inf here
    it_transitions = iter(block_transitions)
    
    departure = next(it_departures)
    arrival = next(it_arrivals)
    transition = next(it_transitions)
    
    p_action = np.zeros(n_bins)
    p_leaving = np.zeros(n_bins)
       
    for i in range(1, n_bins+1):
        t = round(i * time_bin, 1)
        
        #check current block type and select corresponding threshold
        if t > transition:
            transition = next(it_transitions)
            if bias == bias_short:
                bias = bias_long
            elif bias == bias_long:
                bias = bias_short
        
        # check if the upcoming reward is in this time bin, and if so start looking out for the next reward
        
        if next_reward_index < n_rewards:
            r = 0 if reward_times[next_reward_index] > t else 1        
            if r == 1:
                next_reward_index += 1
        else:
            r = 0
        
         # update environment reward rate
        delta_env[i] = r-rho_env[i-1]
        rho_env[i] = rho_env[i-1] + alpha_env * delta_env[i] # if r = 1 use alpha_pos, if equal to 0 use alpha_neg  

        
        #if the animal has stayed in the patch throughout this time bin, update the patch estimates and compute probability of staying after update
        if t < departure:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
            p_action[i-1] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))            
            p_leaving[i-1] = 1 - p_action[i-1]
        
        # if travelling keep rho_patch constant
        elif t >= departure:
            
            if first_bin: #update rho_patch one last time and compute probability of leaving
                delta_patch[i] = r - rho_patch[i-1]
                rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
                p_action[i-1] = np.exp(beta * rho_env[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))
                p_leaving[i-1] = p_action[i-1]                
                first_bin = False
            else:
                delta_patch[i] = 0
                rho_patch[i] = reset
                p_action[i-1] = 1
                p_leaving[i-1] = 0
                
            if t > arrival:
                
                departure = next(it_departures)
                arrival = next(it_arrivals)
                first_bin = True

    return (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving)