# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:11:41 2021

@author: François Cinotti

Improved by Thomas Akam on 20/04/2021 (Numba and store_variables)
"""

import numpy as np
from numba import jit

def ExtractChoices(session, time_bin = 100):
    # converts a session into a vector of 0 and 1 corresponding to the 
    # decision to quit or continue foraging the current patch respectively for 
    # time bins of a given duration 
    complete_duration = FocusedTime(session)
    n_bins = int(complete_duration / time_bin) + 1
    travel_times = FocusedPatchTransitions(session)
    i = 0
    
    choices_vector = np.zeros((n_bins))
    
    for bin_id in range(n_bins):
        t = bin_id * time_bin
        
        if t > travel_times["arrivals"][i]:
            i += 1
        
        if t > travel_times["departures"][i] and t < travel_times["arrivals"][i]:
            choices_vector[bin_id] = 0
        else: 
            choices_vector[bin_id] = 1
            
    return choices_vector.astype(int)
        

    
# def EnvironmentRewardRate(session, alpha = 0.05, time_bin = 100):
#     #reward_times = sorted(np.concatenate((session.times['reward_right'],session.times['reward_left']))) # times reward was collected
#     reward_times = FocusedRewardTimes(session)
#     n_bins = int(FocusedTime(session) / time_bin) + 1
#     rho = np.zeros(n_bins)
#     delta = np.zeros(n_bins)
    
#     next_reward_index = 0
    
#     for i in range(1, n_bins):
#         t = i * time_bin
#         #print(t)
        

#         r = 0 if reward_times[next_reward_index] > t else 1
        
#         if reward_times[next_reward_index] < t:
#             next_reward_index += 1
            
#         #print(r)
#         delta[i] = r-rho[i-1]
#         rho[i] = rho[i-1] + alpha * delta[i] # if r = 1 use alpha_pos, if equal to 0 use alpha_neg      

#     return [rho, delta]

# def EngagedTimeStamps(session):
#     # calculates the time stamps of rewards, patch arrivals and departures
#     # when only periods of engagement (i.e. nosepokes) are considered
#     duration = 0.
    
#     reward_times = np.zeros(session.n_rewards+1)
#     i = 1
#     travel_time = 0
#     give_up_time = 0
    
#     n_patches = len(session.patch_data)
#     patch_arrivals = np.zeros(n_patches+1)
#     patch_departures = np.zeros(n_patches+1)
#     i = 1
    
#     for patch in session.patch_data:
#         duration += np.nansum(np.append(patch["forage_time"], [patch["travel_time"], patch["give_up_time"]]))
#     return duration

def EngagedTimeStamps(session):
    # calculates the time stamps of rewards, patch arrivals and departures
    # when only periods of engagement (i.e. nosepokes) are considered, except
    # for travel time which is the total amount of time between leaving a patch
    # and arriving in a new one
    
    # initialise the outputs, i.e. total session duration, times of patch 
    # arrivals and departures and reward times
    
    total_duration = 0
    
    n_patches = len(session.patch_data)
    patch_arrivals = np.zeros(n_patches+1)
    patch_departures = np.zeros(n_patches+1)
    
    reward_times = np.zeros(session.n_rewards+1)
    
    rwd_idx = 1 #index of rewards between all patches
    
    for patch_idx, patch in enumerate(session.patch_data):
        
        total_duration += np.nansum(np.append(patch["forage_time"], [patch["travel_time"], patch["give_up_time"]]))
        
        patch_departures[patch_idx + 1] = patch_departures[patch_idx] + sum(patch['forage_time']) + patch["give_up_time"]
        patch_arrivals[patch_idx + 1] = patch_departures[patch_idx+1] + patch["travel_time"]
        
        for rwd_patch_idx in range(len(patch["forage_time"])): # looping over rewards within current patch
            # time of current reward is time of arrival in the current patch plus the amount of time spent
            # foraging for this and previous rewards inthe same patch
            reward_times[rwd_idx] = patch_arrivals[patch_idx] + sum(patch["forage_time"][:rwd_patch_idx + 1])

            rwd_idx += 1
    
    return (total_duration, reward_times[1:], patch_departures[1:], patch_arrivals[1:])
    

def FocusedRewardTimes(session):
    reward_times = np.zeros(session.n_rewards+1)
    i = 1
    travel_time = 0
    give_up_time = 0
    for patch in session.patch_data:
        for in_patch_idx in range(len(patch["forage_time"])):
            reward_times[i] = reward_times[i-1] + patch["forage_time"][in_patch_idx] + (in_patch_idx == 0) * (travel_time + give_up_time)
            if in_patch_idx == len(patch["forage_time"]) - 1:
                travel_time = patch["travel_time"]
                give_up_time = patch["give_up_time"]
            i += 1
    return reward_times[1:]

def FocusedPatchTransitions(session):
    n_patches = len(session.patch_data)
    patch_arrivals = np.zeros(n_patches+1)
    patch_departures = np.zeros(n_patches+1)
    i = 1
    for patch in session.patch_data:
        patch_departures[i] = patch_arrivals[i-1] + sum(patch['forage_time']) + patch["give_up_time"]
        patch_arrivals[i] = patch_departures[i] + patch["travel_time"]
        i += 1
    return {"departures": patch_departures[1:], "arrivals": patch_arrivals[1:]}

def PatchRewardRate(session, alpha = 0.2, offset = 0.2,  time_bin = 100):
    reward_times = FocusedRewardTimes(session)
    travel_times = FocusedPatchTransitions(session)
    n_bins = int(FocusedTime(session) / time_bin) + 1
    
    rho = np.zeros(n_bins)
    delta = np.zeros(n_bins)
    
    next_reward_index = 0
    patch_idx = 0
    
    for i in range(1, n_bins):
        t = i * time_bin
        
        # if travelling keep patch rho constant
        
        if t > travel_times["departures"][patch_idx] and t < travel_times["arrivals"][patch_idx]:
            delta[i] = 0
            rho[i] = rho[i-1]
        
        # when arriving in a new patch reset rho
        
        elif t > travel_times["arrivals"][patch_idx]:
            delta[i] = offset
            rho[i] = offset
            patch_idx += 1
        else:
            r = 0 if reward_times[next_reward_index] > t else 1
            
            delta[i] = r - rho[i-1]
            rho[i] = rho[i-1] + alpha * delta[i]
            if r == 1:
                next_reward_index += 1
    return [rho, delta]

@jit(nopython=True)
def RewardRates(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, n_bins, average_reward_rate):
    '''Changes made to make fuction compatible with numba.jit:
    - All arguments and returned values are now either numpy arrays or
      individual numbers.
    - p_action array is now pre-allocated rather than appended inside loop.
    '''
    # unpack model parameters.
    alpha_env, alpha_patch, beta, offset, bias = params

    rho_env = np.zeros(n_bins)
    rho_env[0] = average_reward_rate
    delta_env = np.zeros(n_bins)
    
    rho_patch = np.zeros(n_bins)
    rho_patch[0] = offset
    delta_patch = np.zeros(n_bins)
    
    next_reward_index = 0
    patch_idx = 0
    first_bin = True
    nd = 0 # Counter for number of decison points.
    
    p_action = np.zeros(n_bins)
    
    for i in range(1, n_bins):
        t = i * time_bin
        
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
        
        #if in a patch update rho_patch in the same way, and calculate the probability that the animal decided to stay based on reward rates at the preceding time step
        if t < departure_times[patch_idx]:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
            p_action[nd] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))
            nd += 1
        
        # if travelling keep rho_patch constant
        elif t > departure_times[patch_idx]:
            delta_patch[i] = 0
            rho_patch[i] = offset
            if first_bin:
                p_action[nd] = np.exp(beta * rho_env[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))
                nd += 1
                first_bin = False
            if t > arrival_times[patch_idx]:
                patch_idx += 1
                first_bin = True

    p_action = p_action[:nd]
    return (rho_env, delta_env, rho_patch, delta_patch, p_action)


def store_variables(session, time_bin=100):
    '''Compute the variables needed for the RL model and store them
    on the session.'''
    travel_times = FocusedPatchTransitions(session)
    n_bins = int(FocusedTime(session) / time_bin) + 1
    session.model_vars = {
        'reward_times'        : FocusedRewardTimes(session),
        'departure_times'     : travel_times["departures"],
        'arrival_times'       : travel_times['arrivals'],
        'n_bins'              : n_bins,
        'time_bin'            : time_bin,
        'n_rewards'           : session.n_rewards,
        'average_reward_rate' : session.n_rewards / n_bins}


def LogLikelihood(params, session):
    '''Calculate the LogLikelihood for given parameters and session.  The 
    session must first have been passed to the store_variables function 
    to pre-calculate variables and store them on the session.'''
    (rho_env, delta_env, rho_patch, delta_patch, p_action) = RewardRates(params, **session.model_vars)
    loglikelihood = -1 * np.sum(np.log(p_action))
    return loglikelihood