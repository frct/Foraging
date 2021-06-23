# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:03:31 2021

Aim: go through the raw text file line by line and extract foraging and
travelling decisions, instead of relying on print lines

@author: franc
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def ExtractPatches(file_path):
    with open(file_path, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    info_lines = [line[2:] for line in all_lines if line[0]=='I']
    subject_ID_string = next(line for line in info_lines if 'Subject ID' in line).split(' : ')[1]
    subject_ID = int(''.join([i for i in subject_ID_string if i.isdigit()]))
    datetime_string = next(line for line in info_lines if 'Start date' in line).split(' : ')[1]
    session_date = datetime.strptime(datetime_string, '%Y/%m/%d %H:%M:%S').date()
    
    data_lines = [line.split(' ') for line in all_lines if line[0] in ['D', 'P']] # print and data lines

    
    state_IDs = eval(next(line for line in all_lines if line[0]=='S')[2:])
    event_IDs = eval(next(line for line in all_lines if line[0]=='E')[2:])
    
    ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}
    
    patch_number = 0
    foraging_bouts = []
    patch_data = [{'forage_times':[], 'forage_before_travel': [], 'travel_bouts': [], 'reward times':[], 'complete_travel_time': [], 'start': 0, 'patch number': 1}]
    
    # initialise ids
    start_forage_id = []
    stop_forage_id = []
    reward_available_id = []
    reward_consumption_id = []
    
    reward = False
    travelling = False
    
    for line in data_lines:
        #print(line[1])
        if line[0] == 'P':
            if 'TT:1000' in line:
                block = 'short'
            elif 'TT:4000' in line:
                block = 'long'
            elif 'IFT:227.583' in line:
                richness = 'rich'
            elif 'IFT:500' in line:
                richness = 'medium'
            elif 'IFT:1098.5' in line:
                richness = 'poor'
        
        else:   
        
            # identify the correct events for current patch   
            if ID2name[int(line[2])] == 'start_forage_left':
                #print('forage left')
                start_forage_id = event_IDs['poke_2']
                stop_forage_id = event_IDs['poke_2_out']
                reward_available_id = state_IDs['reward_left_available']
                reward_consumption_id = state_IDs['reward_consumption_left']
                patch_data[patch_number]['forage poke'] = 'left'
                
            elif ID2name[int(line[2])] == 'start_forage_right':
                #print('forage right')       
                start_forage_id = event_IDs['poke_3']
                stop_forage_id = event_IDs['poke_3_out']
                reward_available_id = state_IDs['reward_right_available']
                reward_consumption_id = state_IDs['reward_consumption_right']
                patch_data[patch_number]['forage poke'] = 'right'
            
            if not reward and not travelling: # record foraging events only if no reward is currently available
                if int(line[2]) == start_forage_id:
                    start_time = int(line[1])
                    
                if int(line[2]) == stop_forage_id or int(line[2]) == reward_available_id:
                    stop_time = int(line[1])
                    foraging_bouts.append([start_time, stop_time])
                    if int(line[2]) == reward_available_id: # if a reward is available, then the current trial is over and the foraging bouts are added to trial
                        patch_data[patch_number]["forage_times"].append(foraging_bouts)
                        start_reward = int(line[1])
                        foraging_bouts = []
                        reward = True
            elif int(line[2]) == reward_consumption_id: # if reward is eaten, start recording foraging events again
                reward = False
                end_reward = int(line[1])
                patch_data[patch_number]["reward times"].append([start_reward, end_reward])
            
            if ID2name[int(line[2])] == 'travel' and not travelling: #sometimes, the travel event resets when an animal has lost interest, not travelling condition ensures these repetitions are simply concatenated with the previous travel
                travelling = True
                travel_start = int(line[1])
                
                if foraging_bouts: #if the animal has foraged without collecting a reward before starting travel, record this apart and reset trial_foraging_bouts
                    patch_data[patch_number]["forage_before_travel"] = foraging_bouts
                    foraging_bouts = []
            
            if ID2name[int(line[2])] == 'poke_9':
                travel_bout_start = int(line[1])
                in_poke_9 = True
        
            if ID2name[int(line[2])] == 'poke_9_out':
                travel_bout_end = int(line[1])
                in_poke_9 = False
                if travelling : # if still travelling, this bout adds into the current patch travel, otherwise just ignore it
                    patch_data[patch_number]['travel_bouts'].append([travel_bout_start, travel_bout_end])    
                
            if ID2name[int(line[2])] == 'travel_complete':
                travel_end = int(line[1])
                
                if in_poke_9: #if still in a travel poke bout, add this last bout
                    patch_data[patch_number]['travel_bouts'].append([travel_bout_start, travel_end])
                
                patch_data[patch_number]['complete_travel_time'] = [travel_start, travel_end]
                patch_data[patch_number]['block'] = block
                patch_data[patch_number]['richness'] = richness
                patch_data[patch_number]['end'] = travel_start
                travelling = False
                patch_number += 1
                patch_data.append({'forage_times':[], 'forage_before_travel': [], 'travel_bouts': [], 'reward times': [], 'complete_travel_time': [], 'start': travel_end, 'patch number': patch_number + 1})
    
    patch_data[-1]['block'] = patch_data[-2]['block']
    patch_data[-1]['end'] = int(line[1])
    patch_data[-1]['richness'] = richness
    patch_data
    
    return (patch_data, subject_ID, session_date)

file_path = '../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt'
first_patch = 0
last_patch = None

def PlotSessionTimeCourse(file_path, first_patch = 0, last_patch = None):
    all_patches, subject, session = ExtractPatches(file_path)
    
    selection = slice(first_patch, last_patch)
    patches = all_patches[selection]
    
    
    reward_times = []
    
    left_forage = []
    right_forage = []
    
    poor_patch = []
    medium_patch = []
    rich_patch = []
    
    
    
    short_blocks = []
    long_blocks = []
    
    for n, patch in enumerate(patches):
        
        
        if patch['forage poke'] == 'left':
            left_forage.append([patch['start'], patch['end']])
        elif patch['forage poke'] == 'right':
            right_forage.append([patch['start'], patch['end']])
        else:
            print('Error: no poke information')
            
        if patch['richness'] == 'poor':
            poor_patch.append([patch['start'], patch['end']])
        elif patch['richness'] == 'medium':
            medium_patch.append([patch['start'], patch['end']])
        elif patch['richness'] == 'rich':
            rich_patch.append([patch['start'], patch['end']])
        else:
            print('Error: no patch richness information')
            
        if patch['block'] == 'short':
            if n == 0:
                short_travel_start = patch['start']
            elif patches[n-1]['block'] == 'long':
                short_travel_start = patch['start']
                long_blocks.append([long_travel_start, short_travel_start])
        elif patch['block'] == 'long':
            if n == 0:
                long_travel_start = patch['start']
            elif patches[n-1]['block'] == 'short':
                long_travel_start = patch['start']
                short_blocks.append([short_travel_start, long_travel_start])
        else:
            print('Error: no block information')
        
        for reward in patch['reward times']:
            reward_times.append(reward[0])
    
        
    
    first_instant = patches[0]['forage_times'][0][0][0]
    last_instant = patches[-1]['reward times'][-1][1]
    time_window = range(first_instant, last_instant+1)
    n_t = len(time_window)
    
    #initiate reward state
    
    reward_state = np.zeros(n_t)
    rwd_nb = 0
    
    # initiate travel block
    
    short_travel_block = np.zeros(n_t)
    long_travel_block = np.zeros(n_t)
    block_nb = 0
    
    # poor patch 
    
    poor_patch_state = np.zeros(n_t)
    poor_patch_nb = 0
    
    medium_patch_state = np.zeros(n_t)
    medium_patch_nb = 0
    
    rich_patch_state = np.zeros(n_t)
    rich_patch_nb = 0
    
    # initialise left or right patch parameters
    
    left_patch_nb = 0
    right_patch_nb = 0
    
    for i,t in enumerate(time_window):
        
        # update block state
        
        if t > short_blocks[block_nb][0] and t < short_blocks[block_nb][1]:
            short_travel_block[i] = 1
        else:
            long_travel_block[i] = 1
        
        if t > short_blocks[block_nb][1] and block_nb < len(short_blocks) - 1:
            block_nb += 1
            
        # #check and update reward state
        if rwd_nb < len(reward_times) - 1:
            if t > reward_times[rwd_nb]:
                reward_state[i] = 1
                rwd_nb += 1
        
        # check left or right patch and code forage state as either 1 (left forage) or -1 (right forage)
        if t > left_forage[left_patch_nb][0] and t < left_forage[left_patch_nb][1]:
            forage_id = 1
        if t >= left_forage[left_patch_nb][1] and left_patch_nb < len(left_forage) - 1:
            left_patch_nb += 1
        if t > right_forage[right_patch_nb][0] and t < right_forage[right_patch_nb][1]:
            forage_id = -1
        if t >= right_forage[right_patch_nb][1] and right_patch_nb < len(right_forage) - 1:
            right_patch_nb += 1
            
        #check and update poor patch
        if t > poor_patch[poor_patch_nb][0] and t < poor_patch[poor_patch_nb][1]:
            poor_patch_state[i] = forage_id
        if t >= poor_patch[poor_patch_nb][1] and poor_patch_nb < len(poor_patch) - 1:
            poor_patch_nb += 1
        
        # check and update medium patch state
        if t > medium_patch[medium_patch_nb][0] and t < medium_patch[medium_patch_nb][1]:
            medium_patch_state[i] = forage_id
        if t >= medium_patch[medium_patch_nb][1] and medium_patch_nb < len(medium_patch) - 1:
            medium_patch_nb += 1
        
        if t > rich_patch[rich_patch_nb][0] and t < rich_patch[rich_patch_nb][1]:
            rich_patch_state[i] = forage_id
        if t >= rich_patch[rich_patch_nb][1] and rich_patch_nb < len(rich_patch) - 1:
            rich_patch_nb += 1
    
    plt.figure(figsize=(20,10))
    
    plt.subplot(3,1,1)
    plt.fill_between(time_window, short_travel_block, 0, 
                      facecolor = 'b',
                      color = 'b',
                      alpha = 0.2)
    plt.fill_between(time_window, long_travel_block, 0,
                      facecolor = 'r',
                      color = 'r',
                      alpha = 0.2)
    plt.title('short and long travel time blocks')
    
    plt.subplot(3,1,2)
    poor = plt.fill_between(time_window, poor_patch_state, 0, 
                      facecolor = 'r',
                      color = 'r',
                      alpha = 0.2,
                      label = 'Poor')
    medium = plt.fill_between(time_window, medium_patch_state, 0,
                      facecolor = 'b',
                      color = 'b',
                      alpha = 0.2, 
                      label = 'medium')
    rich = plt.fill_between(time_window, rich_patch_state, 0,
                      facecolor = 'g',
                      color = 'g',
                      alpha = 0.2,
                      label = 'rich')
    plt.title('Patch sequence')
    plt.yticks([-1, 0, 1], labels = ['right', 'travel', 'left'])
    plt.legend(handles = [poor, medium, rich])
    
    plt.subplot(3,1,3)
    plt.plot(reward_state)
    plt.title('Rewards')
    
    plt.savefig('session %s summary.png' % session, format = 'png')
    
# file_path = '../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt'
# PlotSessionTimeCourse(file_path, first_patch = 0, last_patch=None)

def PlotPatchEvents(filepath):

    patches, subject, date = ExtractPatches(file_path)
    
    for p, patch in enumerate(patches):
        
        first_instant = patch['start']
        last_instant = patch['end']
        
        margin = 100
        
        first_instant = patch['forage_times'][0][0][0]
        if patch['forage_before_travel']:
            last_instant = patch['forage_before_travel'][-1][1]
        else:
            last_instant = patch['reward times'][-1][1]
        
        in_patch_time_window = range(first_instant - margin, last_instant + margin)
        n_t = len(in_patch_time_window)
        
        # flatten foraging bouts and include forage_before_travel bouts
        
        foraging_bouts = []
        for trial in patch['forage_times']:
            for bout in trial:
                foraging_bouts.append(bout)
        if patch['forage_before_travel']:
            for bout in patch['forage_before_travel']:
                foraging_bouts.append(bout)
        
        forage_state = np.zeros(n_t)
        reward_state = np.zeros(n_t)
        
        rwd_nb = 0
        bout = 0
        
        for i,t in enumerate(in_patch_time_window):
            
            if t > foraging_bouts[bout][0] and t < foraging_bouts[bout][1]:
                forage_state[i] = 1
            
            if t > foraging_bouts[bout][1] and bout < len(foraging_bouts) - 1:
                bout += 1
                
            if rwd_nb < len(patch['reward times']): #if any rewards are left, keep an eye out for the next one
                if t > patch['reward times'][rwd_nb][0] :
                    reward_state[i] = 1
                    rwd_nb += 1
    
        plt.figure()
        plt.plot(in_patch_time_window, forage_state)
        plt.plot(in_patch_time_window, reward_state, 'r')
        plt.title(patch['richness'] + ' patch number ' + str(p+1))
        plt.xlabel('Time (ms)')
        plt.tick_params(left = False, labelleft = False)
        plt.savefig('mouse %s/session %s/patch %s' %(subject, date,  patch['patch number']), format = 'pdf')
        
        if patch['travel_bouts']:
            travel_start = patch['travel_bouts'][0][0]
            travel_end = patch['travel_bouts'][-1][1]
            travel_time_window = range(travel_start - margin, travel_end + margin)
            n_t = len(travel_time_window)
            
            travel_state = np.zeros(n_t)
            poke = 0
            
            for i, t in enumerate(travel_time_window):
                if t > patch['travel_bouts'][poke][0] and t < patch['travel_bouts'][poke][1]:
                    travel_state[i] = 1
                
                if t > patch['travel_bouts'][poke][1] and poke < len(patch['travel_bouts']) - 1:
                    poke += 1
            
            plt.figure()
            plt.plot(travel_time_window, travel_state)
            plt.title(patch['block'] + ' travel pokes between patch %s and %s' %(p+1, p+2))
            plt.xlabel('Time (ms)')
            plt.tick_params(left = False, labelleft = False)
            plt.savefig('mouse %s/session %s/patch %s travel' %(subject, date,  patch['patch number']), format = 'pdf')
    
# file_path = '../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt'
# PlotPatchEvents(file_path)