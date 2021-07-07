# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:11:41 2021

@author: François Cinotti

Improved by Thomas Akam on 20/04/2021 (Numba and store_variables)
"""

import numpy as np
import analyse_behaviour as analysis
import import_pycontrol as ip
import SingleSessionAnalysis as ssa
from numba import jit

# def ExtractChoices(session, time_bin = 100):
#     # converts a session into a vector of 0 and 1 corresponding to the 
#     # decision to quit or continue foraging the current patch respectively for 
#     # time bins of a given duration 
#     complete_duration = FocusedTime(session)
#     n_bins = int(complete_duration / time_bin) + 1
#     travel_times = FocusedPatchTransitions(session)
#     i = 0
    
#     choices_vector = np.zeros((n_bins))
    
#     for bin_id in range(n_bins):
#         t = bin_id * time_bin
        
#         if t > travel_times["arrivals"][i]:
#             i += 1
        
#         if t > travel_times["departures"][i] and t < travel_times["arrivals"][i]:
#             choices_vector[bin_id] = 0
#         else: 
#             choices_vector[bin_id] = 1
            
#     return choices_vector.astype(int)


def CompressTime(session_summary, travel_engagement):
    # calculates the time stamps of rewards, patch arrivals and departures
    # when only periods of engagement (i.e. nosepokes) are considered

    n_patches = session_summary['number of patches']
    N_rewards = sum(session_summary['rewards per patch'])
    
    #initialisation of outputs
    
    total_duration = 0
    patch_departures = np.zeros(n_patches)
    patch_arrivals = np.zeros(n_patches)
    reward_times = np.zeros(N_rewards)
    
    # initialise counters
    
    last_departure = 0
    last_arrival = 0
    rwd_idx = 0
    
    for patch_id in range(n_patches):
        if travel_engagement: #only consider time in nosepokes
            if patch_id + 1 < n_patches:
                travel_time = session_summary['total time in travel poke'][patch_id]
            else:
                travel_time = 0
        else:
            if patch_id + 1 < n_patches:
                travel_time = session_summary['duration of travel'][patch_id]
            else:
                travel_time = 0
                
        foraging_time = session_summary['total successful forage time per patch'][patch_id] + session_summary['give up time'][patch_id]
        total_duration += travel_time + foraging_time
        
        last_departure = last_arrival + foraging_time
        last_arrival = last_departure + travel_time
        patch_departures[patch_id] = last_departure
        
        if patch_id + 1 < n_patches:
            patch_arrivals[patch_id+1] = last_arrival
        rewards = patch_arrivals[patch_id] + np.cumsum(session_summary['forage time for each reward'][patch_id])
        
        for rwd in rewards:
            reward_times[rwd_idx] = rwd
            rwd_idx += 1
    return {"total duration": total_duration, "reward times": reward_times, "patch departures": patch_departures, "patch arrivals": patch_arrivals}


@jit(nopython=True)
def RewardRates(params, n_rewards, time_bin, reward_times, departure_times,
                arrival_times, n_bins, average_reward_rate):
    '''Changes made to make fuction compatible with numba.jit:
    - All arguments and returned values are now either numpy arrays or
      individual numbers.
    - p_action array is now pre-allocated rather than appended inside loop.
    '''
    # unpack model parameters.
    alpha_env, alpha_patch, beta, reset, bias = params

    rho_env = np.zeros(n_bins)
    rho_env[0] = average_reward_rate
    delta_env = np.zeros(n_bins)
    
    rho_patch = np.zeros(n_bins)
    rho_patch[0] = reset
    delta_patch = np.zeros(n_bins)
    
    next_reward_index = 0
    patch_idx = 0
    first_bin = True
    nd = 0 # Counter for number of decison points.
    
    p_action = np.zeros(n_bins)
    p_leaving = np.zeros(n_bins)
    
    for i in range(1, n_bins+1):
        t = i * time_bin
        #print(t)
        
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
        if t < departure_times[patch_idx]:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
            p_action[nd] = np.exp(bias + beta * rho_patch[i-1]) / (np.exp(bias + beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1]))            
            p_leaving[nd] = 1 - p_action[nd]
            nd += 1
        
        # if travelling keep rho_patch constant
        elif t > departure_times[patch_idx]:
            
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
                
            if t > arrival_times[patch_idx]:

                patch_idx += 1
                first_bin = True

    p_action = p_action[:nd]
    p_leaving = p_leaving[:nd]
    return (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving)


def PrepareForOptimisation(session, time_bin=100, Engaged = True):
    '''Compute the variables needed for the RL model and store them
    on the session.'''
    
    session_summary = ssa.SummaryMeasures(session)
    
    time_stamps = CompressTime(session_summary, Engaged)
    n_rewards = sum(session_summary['rewards per patch'])
    
    n_bins = int(time_stamps["total duration"] / time_bin) + 1
    model_vars = {
        'reward_times'        : time_stamps["reward times"],
        'departure_times'     : time_stamps["patch departures"],
        'arrival_times'       : time_stamps["patch arrivals"],
        'n_bins'              : n_bins,
        'time_bin'            : time_bin,
        'n_rewards'           : n_rewards,
        'average_reward_rate' : n_rewards / n_bins}
    
    return model_vars

# def ExtractGiveUpTime(session):
#     file_path = '../../raw_data/behaviour_data/' + session.file_name
    
#     with open(file_path, 'r') as f:
#         all_lines = [line.strip() for line in f.readlines() if line.strip()]
    
#     print_lines = [line[2:].split(' ',1) for line in all_lines if line[0]=='P'] 
#     data_lines = [line[2:].split(' ') for line in all_lines if line[0]=='D']
#     trial_lines = [line for line in print_lines if 'P:' in line[1]] # Lines with trial data.
    
#     patch_departure_lines = [line for i,line in enumerate(trial_lines) if 'P:-1' in line[1] and 'P:-1' not in trial_lines[i-1][1]]
#     last_reward_in_patch_lines = [line for i, line in enumerate(trial_lines[:-1]) if 'P:-1' not in line[1] and 'P:-1' in trial_lines[i+1][1]]
    
#     start_foraging_lines = [line for line in data_lines if line[1] in ['19', '21']]
#     stop_foraging_lines = [line for line in data_lines if line[1] in ['20', '22']]
    
#     patch = 0
#     give_up_time = np.zeros((session.n_patches))
    
#     for i, (start, stop) in enumerate(zip(start_foraging_lines, stop_foraging_lines)):
#         if patch < session.n_patches - 1:
#             if int(start[0]) > int(last_reward_in_patch_lines[patch][0]) and int(start[0]) < int(patch_departure_lines[patch][0]):
#                 give_up_time[patch] += int(stop[0]) - int(start[0]) 
#             if int(start[0]) > int(patch_departure_lines[patch][0]):
#                 patch += 1
            
#     session.corrected_give_up_time = give_up_time
    
def LogLikelihood(params, model_vars):
    '''Calculate the LogLikelihood for given parameters and session.  The 
    session must first have been passed to the store_variables function 
    to pre-calculate variables and store them on the session.'''
    (rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving) = RewardRates(params, **model_vars)
    loglikelihood = -1 * np.sum(np.log(p_action))
    return loglikelihood