# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:21:15 2021

@author: franc
"""

import numpy as np
import os
import ProcessRawData as ssa
import random
import pickle
from itertools import cycle
import matplotlib.pyplot as plt
import ModelOptimisation as opt


def GiveReward(forage_time, required_time):
    if forage_time >= required_time:
        rwd = 1
    else:
        rwd = 0
    return rwd

def GetTravelDurations(mouse):
    # get all sessions of this individual
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    files = os.listdir(folder_path)
    
    long_travels = []
    short_travels = []
    
    for session_idx, file in enumerate(files):
        filepath = folder_path + '/' + file
        patches, _, _ = ssa.ExtractPatches(filepath, 'ms')
        des = ssa.SummaryMeasures(patches)

        
        for patch_idx, patch in enumerate(patches[:-1]): #last patch doesn't have a travel time
            travel_duration = des['total time in travel poke'][patch_idx]
            if patch['block'] == 'long':
                long_travels.append(travel_duration)
            elif patch['block'] == 'short':
                short_travels.append(travel_duration)
            else:
                raise Exception('block length not recognised')
    
    travel_durations = {'long': long_travels, 'short': short_travels}         
    return travel_durations

def GetAverageRewardRate(mouse):
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    files = os.listdir(folder_path)
    
    n_rewards = 0
    n_bins = 0
    
    for session_idx, file in enumerate(files):
        filepath = folder_path + '/' + file
        this_session, subject, date = ssa.ExtractPatches(filepath, 's')
        session_vars = opt.PrepareForOptimisation(this_session, time_bin = 0.1, Engaged = True)
        n_rewards += session_vars['n_rewards']
        n_bins += session_vars['n_bins']
    avg = n_rewards / n_bins
    
    return avg

    

def GenerateSessionStructure(mouse):
    
    types = ['long', 'short']
    random.shuffle(types) #randomly start session with either a short or a long travel block
    block_type = cycle(types) #generate a cycle object that toggles between short and long
    
    blocks = []
    n_min = 40
    
    while len(blocks) < n_min:
        block = next(block_type)
        blocks.extend([block] * random.randint(8,12))
        
    patch_types = ['rich'] * 2 + ['medium'] * 2 + ['poor'] * 2
    initial_avg_forage = {'rich': 500 / (1.3**3), 'medium': 500, 'poor': 500 * 1.3**3}
    random.shuffle(patch_types)
    patch_iter = iter(patch_types)
    
    richness = []
    
    for patch in blocks:
        patch_richness = next(patch_iter, None)
        if patch_richness == None:
            random.shuffle(patch_types)
            patch_iter = iter(patch_types)
            patch_richness = next(patch_iter)
        
        
        richness.append(initial_avg_forage[patch_richness])
        
    travel_durations = GetTravelDurations(mouse)
    
    travels = []
    
    for patch in blocks[:-1]:
        if patch == 'long':
            travels.append(random.choice(travel_durations['long']))
        elif patch == 'short':
            travels.append(random.choice(travel_durations['short']))
        else:
            raise Exception('Block type not recognised')
    
    return blocks, richness, travels

def ActionSelection(patch, env, beta, bias):
    p_foraging = np.exp(bias + beta * patch) / (np.exp(bias + beta * patch) + np.exp(beta * env))
    if random.random() < p_foraging:
        action = 'forage'
    else:
        action = 'leave'
    return action, 1-p_foraging

def SimulatePatch(richness, parameters, rho_env=None, rho_patch=None, delta_env = None, delta_patch = None, choices = None, p_leaving = None):
    
    alpha_env, alpha_patch, beta, reset, bias = parameters
    
    avg_forage = richness
    forage_time = random.expovariate(1 / avg_forage)
    remaining_time = forage_time
    if rho_env == None:
         rho_env = []
         rho_patch = []
         delta_env = []
         delta_patch = []
         choices = []
         p_leaving = []
         
    
    # rho_patch.append(reset)
    # rho_env.append(0)
    action = 'forage'
    reward_times = []
    
    while action == 'forage':
        if remaining_time > 0:
            r = 0
        else:
            r = 1
            reward_times.append(forage_time)
            avg_forage *= 1.3
            forage_time = random.expovariate(1 / avg_forage)
            remaining_time = forage_time
        
        delta_env.append(r-rho_env[-1])
        rho_env.append(rho_env[-1] + alpha_env * delta_env[-1])
        delta_patch.append(r-rho_patch[-1])
        rho_patch.append(rho_patch[-1] + alpha_patch * delta_patch[-1])
        
        action, p_l = ActionSelection(rho_patch[-1], rho_env[-1], beta, bias)
        choices.append(action)
        p_leaving.append(p_l)
        remaining_time -= 100
    else:
        give_up_time = forage_time - remaining_time
        simulated_patch = {'giving up time': give_up_time, 'foraging durations': reward_times}
        if richness == 500:
            simulated_patch['richness'] = 'medium'
        elif richness > 1000:
            simulated_patch['richness'] = 'poor'
        else:
            simulated_patch['richness'] = 'rich'
        
        simulated_vars = {'rho_env': rho_env,
                          'rho_patch': rho_patch,
                          'delta_env': delta_env,
                          'delta_patch': delta_patch,
                          'choices': choices,
                          'p_leaving': p_leaving}
        return simulated_vars, simulated_patch

def SimulateTravel(travel_time, parameters, rho_env, rho_patch, delta_env, delta_patch, choices, p_leaving):
    
    alpha_env = parameters[0]
    reset = parameters[3]
    
    while travel_time > 0:
        delta_patch.append(0)
        rho_patch.append(reset)
        delta_env.append(-rho_env[-1])
        rho_env.append(rho_env[-1] + alpha_env * delta_env[-1])
        choices.append('travel')
        p_leaving.append('travel')
        travel_time -= 100
    
    simulated_vars = {'rho_env': rho_env,
                      'rho_patch': rho_patch,
                      'delta_env': delta_env,
                      'delta_patch': delta_patch,
                      'choices': choices,
                      'p_leaving': p_leaving} 
    
    return simulated_vars

       
def SimulateSession(mouse):
    blocks, richness, travels = GenerateSessionStructure(mouse)
    opt_res = pickle.load(open('Model optimisations/Reward rate comparison/Between sessions optimisations/optimisation results for mouse ' + str(mouse) + '.p', 'rb'))
    parameters = opt_res['best result']
    simulated_vars = {'rho_env':[GetAverageRewardRate(mouse)],
                  'delta_env': [],
                  'rho_patch': [parameters[3]],
                  'delta_patch': [],
                  'choices': [],
                  'p_leaving': []}

    
    simulated_patches = []
    
    for patch_nb, block in enumerate(blocks):
        simulated_vars, sim_patch = SimulatePatch(richness[patch_nb], parameters, **simulated_vars)
        sim_patch['block'] = block
        
        if patch_nb < len(travels):
            travel_duration = travels[patch_nb]
            sim_patch['travel duration'] = travel_duration
            simulated_vars = SimulateTravel(travel_duration, parameters, **simulated_vars)
        
        simulated_patches.append(sim_patch)
    
    return simulated_vars, simulated_patches

def SimulateExperiment(mice = [1,2,3,7,8,10,11,12,13,14,16,17]):
    sim_sessions = {'mouse ' + str(mouse): [] for mouse in mice}
    for mouse in mice:
        folder_path = 'mouse ' + str(mouse) + '/raw data'
        files = os.listdir(folder_path)
        n_sessions = len(files)
        simulation = []
        for i in range(n_sessions):
            sim_vars, sim_patches = SimulateSession(mouse)
            sim_sessions['mouse ' + str(mouse)].append({'model variables': sim_vars, 'patches': sim_patches})
            simulation.append({'model variables': sim_vars, 'patches': sim_patches})
        
        pickle.dump(simulation, open('mouse %s/unconstrained simulation.p' %(mouse), 'wb'))
    
    return sim_sessions

def PlotRewardRates(env, patch):
    plt.figure()
    plt.plot(env)
    plt.plot(patch)
    plt.legend(['environment', 'patch'])
    plt.xlabel('time (0.1 s)')
    
    plt.savefig('Simulation.png')