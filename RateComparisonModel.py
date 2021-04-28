# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:11:41 2021

@author: franc
"""

import numpy as np

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
        

    
def EnvironmentRewardRate(session, alpha = 0.05, time_bin = 100):
    #reward_times = sorted(np.concatenate((session.times['reward_right'],session.times['reward_left']))) # times reward was collected
    reward_times = FocusedRewardTimes(session)
    n_bins = int(FocusedTime(session) / time_bin) + 1
    rho = np.zeros(n_bins)
    delta = np.zeros(n_bins)
    
    next_reward_index = 0
    
    for i in range(1, n_bins):
        t = i * time_bin
        #print(t)
        

        r = 0 if reward_times[next_reward_index] > t else 1
        
        if reward_times[next_reward_index] < t:
            next_reward_index += 1
            
        #print(r)
        delta[i] = r-rho[i-1]
        rho[i] = rho[i-1] + alpha * delta[i] # if r = 1 use alpha_pos, if equal to 0 use alpha_neg      

    return [rho, delta]

def FocusedTime(session):
    duration = 0.
    for patch in session.patch_data:
        duration += np.nansum(np.append(patch["forage_time"], [patch["travel_time"], patch["give_up_time"]]))
    return duration

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

def RewardRates(session, alpha_patch = 0.2, alpha_env = 0.05, offset = 0.2,  beta = 2, time_bin = 100):
    reward_times = FocusedRewardTimes(session)
    travel_times = FocusedPatchTransitions(session)
    n_bins = int(FocusedTime(session) / time_bin) + 1
    
    rho_env = np.zeros(n_bins)
    delta_env = np.zeros(n_bins)
    
    rho_patch = np.zeros(n_bins)
    delta_patch = np.zeros(n_bins)
    
    next_reward_index = 0
    patch_idx = 0
    first_bin = True
    
    p_action = np.array((0))
    p_foraging = np.array((0))
    
    for i in range(1, n_bins):
        t = i * time_bin
        
        #check if the upcoming reward is in this time bin, and if so start looking out for the next reward
        
        if next_reward_index < session.n_rewards:
            r = 0 if reward_times[next_reward_index] > t else 1
        
            if r == 1:
                next_reward_index += 1
        else:
            r = 0
                
            
        # update environment reward rate
        delta_env[i] = r-rho_env[i-1]
        rho_env[i] = rho_env[i-1] + alpha_env * delta_env[i] # if r = 1 use alpha_pos, if equal to 0 use alpha_neg  
        
        #if in a patch update rho_patch in the same way, and calculate the probability that the animal decided to stay based on reward rates at the preceding time step
        if t < travel_times["departures"][patch_idx]:
            delta_patch[i] = r - rho_patch[i-1]
            rho_patch[i] = rho_patch[i-1] + alpha_patch * delta_patch[i]
            p_foraging = np.append(p_foraging, np.exp(beta * rho_patch[i-1]) / (np.exp(beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1])))
            p_action = np.append(p_action, np.exp(beta * rho_patch[i-1]) / (np.exp(beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1])))
        
        # if travelling keep rho_patch constant
        elif t > travel_times["departures"][patch_idx]:
            delta_patch[i] = 0
            rho_patch[i] = offset
            if first_bin:
                p_action = np.append(p_action, np.exp(beta * rho_env[i-1]) / (np.exp(beta * rho_patch[i-1]) + np.exp(beta * rho_env[i-1])))
                first_bin = False
            if t > travel_times["arrivals"][patch_idx]:
                patch_idx += 1
                first_bin = True

    return [rho_env, delta_env, rho_patch, delta_patch, p_action[1:], p_foraging[1:]]

def SoftmaxRewardRateComparison(patch_rho, environment_rho, beta):
    lst = [np.exp(beta*p) / (np.exp(beta*p) + np.exp(beta * e)) for p,e in zip(patch_rho, environment_rho)]
    forage_probability = np.array(lst)
    #print(forage_probability)
    return forage_probability

def LogLikelihood(p, session, dt):
    alpha_env = p[0]
    alpha_patch = p[1]
    offset = p[2]
    beta = p[3]
    # [patch_rho, _] = PatchRewardRate(session, alpha_patch, reset, dt)
    # [env_rho, _] = EnvironmentRewardRate(session, alpha_env, dt)
    # choices = ExtractChoices(session)
    # p_foraging = SoftmaxRewardRateComparison(patch_rho, env_rho, beta)
    [rho_env, delta_env, rho_patch, delta_patch, p_action, p_foraging] = RewardRates(session, alpha_patch, alpha_env, offset, beta, dt)
    # likelihood = np.array([p if c == 1 else 1-p for p,c in zip(p_foraging, choices)])
    # print(likelihood)
    loglikelihood = -1 * np.sum(np.log(p_action))
    return loglikelihood