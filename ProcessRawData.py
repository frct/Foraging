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

def ExtractPatches(file_path, unit):
    with open(file_path, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if unit == 's':
        dt = 0.001 # by default, raw data is in ms
    elif unit == 'ms':
        dt = 1
    else:
        print('unit error')
        
    
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
    patch_data = [{'forage times':[], 'forage before travel': [], 'travel bouts': [], 'reward times':[], 'complete travel time': [], 'start': 0, 'patch number': 1}]
    
    # initialise ids
    start_forage_id = []
    stop_forage_id = []
    reward_available_id = []
    reward_consumption_id = []
    
    reward = False
    travelling = False
    
    for line in data_lines:
        if line[0] == 'P':
            if 'TT:1000' in line:
                block = 'short'
            elif 'TT:4000' in line:
                block = 'long'
            elif 'IFT:227.583' in line:
                richness = 'rich'
            elif 'IFT:227.5831' in line:
                richness = 'rich'
            elif 'IFT:500' in line:
                richness = 'medium'
            elif 'IFT:1098.5' in line:
                richness = 'poor'
        
        else:   
        
            # identify the correct events for current patch   
            if ID2name[int(line[2])] == 'start_forage_left':
                
                ''' previous version used:
                start_forage_id = event_IDs['poke_2']
                stop_forage_id = event_IDs['poke_2_out']'''
                
                start_forage_id = state_IDs['in_left']
                stop_forage_id = state_IDs['out_left']
                reward_available_id = state_IDs['reward_left_available']
                reward_consumption_id = state_IDs['reward_consumption_left']
                patch_data[patch_number]['forage poke'] = 'left'
                
            elif ID2name[int(line[2])] == 'start_forage_right':
                
                ''' previous version used:
                start_forage_id = event_IDs['poke_3']
                stop_forage_id = event_IDs['poke_3_out']'''
                
                start_forage_id = state_IDs['in_right']
                stop_forage_id = state_IDs['out_right']
                reward_available_id = state_IDs['reward_right_available']
                reward_consumption_id = state_IDs['reward_consumption_right']
                patch_data[patch_number]['forage poke'] = 'right'
            
            if not reward and not travelling: # record foraging events only if no reward is currently available
                if int(line[2]) == start_forage_id:
                    start_time = round(int(line[1]) * dt, 3)
                    
                if int(line[2]) == stop_forage_id or int(line[2]) == reward_available_id:
                    stop_time = round(int(line[1]) *dt, 3)
                    
                    #somehow it happens that an "out" event is detected before any "in" event at the beginning of a patch (eg : fp12-2019-06-05-110506 or fp01-2019-02-23-130043 (patch 21))
                    # so check if start_time exists, if not ignore
                    
                    if 'start_time' in locals(): # it may happen that an out event at the very beginning of eg 
                        foraging_bouts.append([start_time, stop_time])
                        del start_time
                        
                    if int(line[2]) == reward_available_id: # if a reward is available, then the current trial is over and the foraging bouts are added to trial
                        patch_data[patch_number]["forage times"].append(foraging_bouts)
                        start_reward = round(int(line[1]) * dt, 3)
                        foraging_bouts = []
                        reward = True
                        
            elif int(line[2]) == reward_consumption_id: # if reward is eaten, start recording foraging events again
                reward = False
                end_reward = round(int(line[1]) * dt, 3)
                patch_data[patch_number]["reward times"].append([start_reward, end_reward])
            
            if ID2name[int(line[2])] == 'travel' and not travelling: #sometimes, the travel event resets when an animal has lost interest, not travelling condition ensures these repetitions are simply concatenated with the previous travel
                travelling = True
                travel_start = round(int(line[1]) * dt, 3)
                
                if foraging_bouts: #if the animal has foraged without collecting a reward before starting travel, record this apart and reset trial_foraging_bouts
                    patch_data[patch_number]["forage before travel"] = foraging_bouts
                    foraging_bouts = []
            
            if ID2name[int(line[2])] == 'poke_9':
                travel_bout_start = round(int(line[1]) * dt, 3)
                in_poke_9 = True
        
            if ID2name[int(line[2])] == 'poke_9_out':
                travel_bout_end = round(int(line[1]) * dt, 3)
                in_poke_9 = False
                if travelling : # if still travelling, this bout adds into the current patch travel, otherwise just ignore it
                    patch_data[patch_number]['travel bouts'].append([travel_bout_start, travel_bout_end])    
                
            if ID2name[int(line[2])] == 'travel_complete':
                travel_end = round(int(line[1]) * dt, 3)
                
                if in_poke_9: #if still in a travel poke bout, add this last bout
                    patch_data[patch_number]['travel bouts'].append([travel_bout_start, travel_end])
                
                patch_data[patch_number]['complete travel time'] = [travel_start, travel_end]
                patch_data[patch_number]['block'] = block
                patch_data[patch_number]['richness'] = richness
                patch_data[patch_number]['end'] = travel_start
                travelling = False
                patch_number += 1
                patch_data.append({'forage times':[], 'forage before travel': [], 'travel bouts': [], 'reward times': [], 'complete travel time': [], 'start': travel_end, 'patch number': patch_number + 1})
    
    patch_data[-1]['block'] = patch_data[-2]['block']
    patch_data[-1]['end'] = int(line[1])
    patch_data[-1]['richness'] = richness
    
    return patch_data, subject_ID, session_date

def PlotSessionTimeCourse(file_path):
    patches, subject, date = ExtractPatches(file_path, unit='ms')
    n_patches = len(patches)
    
    margin = 10
    if patches[0]['forage times']:
        first_instant = min(0, patches[0]['forage times'][0][0][0] - margin)
    else:
        first_instant = patches[0]['start']
    if patches[-1]['reward times']:
        last_instant = patches[-1]['reward times'][-1][1] + margin
    else:
        last_instant = patches[-1]['end']
    time_window = range(round(first_instant), round(last_instant))
    n_t = len(time_window)
    
    patch_nb = 0
    rwd_nb = 0
    patch = patches[0]
    n_rwds = len(patch['reward times'])
    
    
    short_travel_block = np.zeros(n_t)
    long_travel_block = np.zeros(n_t)
    
    poor_patch_state = np.zeros(n_t)
    medium_patch_state = np.zeros(n_t)
    rich_patch_state = np.zeros(n_t)
    
    reward_state = np.zeros(n_t)
    
    for i, t in enumerate(time_window):
        
        # update current patch, and reset reward counter
        
        if patch_nb < n_patches - 1: #if already in the last patch don't bother, especially since complete_travel_time is not defined
        
            if t > patches[patch_nb]['complete travel time'][1]:
                patch_nb += 1
                patch = patches[patch_nb]
                n_rwds = len(patch['reward times'])
                rwd_nb = 0
                
            
        
        if patch['forage poke'] == 'left':
            marker = 1
        elif patch['forage poke'] == 'right':
            marker = -1
        else:
            print('Error: missing poke information')
        
        if patch['block'] == 'short':
            short_travel_block[i] = 1
        elif patch['block'] == 'long':
            long_travel_block[i] = 1
        else:
            print('Error: missing block information')
        
        if t > patch['start'] and t < patch['end']: # leave out travelling periods
            if patch['richness'] == 'poor':
                poor_patch_state[i] = marker
            elif patch['richness'] == 'medium':
                medium_patch_state[i] = marker
            elif patch['richness'] == 'rich':
                rich_patch_state[i] = marker
            else:
                print('Error: missing richness information')
                
        if patch['reward times']: #in some patches, the animal doesn't even collect a reward before moving again
            if rwd_nb < n_rwds:
                if t > patch['reward times'][rwd_nb][0]:
                    reward_state[i] = 1
                    rwd_nb += 1

    plt.figure(figsize=(20,10))
    #plt.ioff()
    
    time_window_in_seconds = time_window # [t / 1000 for t in time_window]
    
    plt.subplot(3,1,1)
    short = plt.fill_between(time_window_in_seconds, short_travel_block, 0, 
                      facecolor = 'b',
                      color = 'b',
                      alpha = 0.2,
                      label = 'short travel block')
    long = plt.fill_between(time_window_in_seconds, long_travel_block, 0,
                      facecolor = 'r',
                      color = 'r',
                      alpha = 0.2,
                      label = 'long travel block')
    plt.title('short and long travel time blocks')
    plt.tick_params(left = False, labelleft = False)
    plt.legend(handles = [short, long])
    
    plt.subplot(3,1,2)
    poor = plt.fill_between(time_window_in_seconds, poor_patch_state, 0, 
                      facecolor = 'r',
                      color = 'r',
                      alpha = 0.2,
                      label = 'Poor')
    medium = plt.fill_between(time_window_in_seconds, medium_patch_state, 0,
                      facecolor = 'b',
                      color = 'b',
                      alpha = 0.2, 
                      label = 'medium')
    rich = plt.fill_between(time_window_in_seconds, rich_patch_state, 0,
                      facecolor = 'g',
                      color = 'g',
                      alpha = 0.2,
                      label = 'rich')
    plt.title('Patch sequence')
    plt.yticks([-1, 0, 1], labels = ['right', 'travel', 'left'])
    plt.legend(handles = [poor, medium, rich])
    
    plt.subplot(3,1,3)
    plt.plot(time_window_in_seconds, reward_state)
    plt.tick_params(left = False, labelleft = False)
    plt.title('Rewards')
    plt.xlabel('Time (s)')

    plt.savefig('mouse %s/session %s/timeline/summary.png' %(subject, date), format = 'png')
    plt.close()

def PlotPatchEvents(file_path):
    
    patches, subject, date = ExtractPatches(file_path, unit='ms')
    
    for p, patch in enumerate(patches):
        
        if patch['richness'] == 'poor':
            forage_colour = 'r'
        elif patch['richness'] == 'medium':
            forage_colour = 'b'
        elif patch['richness'] == 'rich':
            forage_colour = 'g'
        
        if patch['block'] == 'short':
            travel_colour = 'b'
        elif patch['block'] == 'long':
            travel_colour = 'r'
        
        #by default use start and end times as boundaries for the figure but if foraging and reward took place focus figure on those events in case of long elapsed time before foraging/after last reward
        first_instant = patch['start']
        last_instant = patch['end']
        
        margin = 100
        
        if patch['forage times']:
            first_instant = patch['forage times'][0][0][0]
            
        if patch['forage before travel']:
            last_instant = patch['forage before travel'][-1][1]
        elif patch['reward times']:
            last_instant = patch['reward times'][-1][1]
        
        in_patch_time_window = range(round(first_instant - margin), round(last_instant + margin))
        n_t = len(in_patch_time_window)
        
        # flatten foraging bouts and include forage_before_travel bouts
        
        foraging_bouts = []
        for trial in patch['forage times']:
            for bout in trial:
                foraging_bouts.append(bout)
        if patch['forage before travel']:
            for bout in patch['forage before travel']:
                foraging_bouts.append(bout)
        
        forage_state = np.zeros(n_t)
        reward_state = np.zeros(n_t)
        
        rwd_nb = 0
        bout = 0
        
        for i,t in enumerate(in_patch_time_window):
            
            if foraging_bouts:
                
                if t > foraging_bouts[bout][0] and t < foraging_bouts[bout][1]:
                    forage_state[i] = 1
                
                if t > foraging_bouts[bout][1] and bout < len(foraging_bouts) - 1:
                    bout += 1
                
            if rwd_nb < len(patch['reward times']): #if any rewards are left, keep an eye out for the next one
                if t > patch['reward times'][rwd_nb][0] and t < patch['reward times'][rwd_nb][1]:
                    reward_state[i] = -1
                if t > patch['reward times'][rwd_nb][1]:
                    rwd_nb += 1
        
        
        
        if patch['travel bouts']:
            travel_start = patch['travel bouts'][0][0]
            travel_end = patch['travel bouts'][-1][1]
            travel_time_window = range(round(travel_start - margin), round(travel_end + margin))
            n_t = len(travel_time_window)
            
            travel_state = np.zeros(n_t)
            poke = 0
            
            for i, t in enumerate(travel_time_window):
                if t > patch['travel bouts'][poke][0] and t < patch['travel bouts'][poke][1]:
                    travel_state[i] = 1
                
                if t > patch['travel bouts'][poke][1] and poke < len(patch['travel bouts']) - 1:
                    poke += 1
            
            time_window_in_seconds = [t / 1000 for t in in_patch_time_window]
            travel_time_window = [t / 1000 for t in travel_time_window]
        
            plt.figure()
            plt.subplot(2,1,1)
            foraging_line, = plt.plot(time_window_in_seconds, forage_state, color = forage_colour, label = 'foraging poke')
            reward_line, = plt.plot(time_window_in_seconds, reward_state, 'orange', label = 'reward available')
            plt.title('patch number ' + str(p+1))
            plt.yticks([-1, 0, 1], labels = ['reward available', 'disengaged', 'in forage poke'])
            
            plt.subplot(2,1,2)
            plt.plot(travel_time_window, travel_state, color = travel_colour)
            plt.xlabel('Time (s)')
            plt.yticks([0, 1], labels = ['disengaged', 'in travel poke'])
            
            plt.savefig('mouse %s/session %s/timeline/patch %s.png' %(subject, date,  patch['patch number']), format = 'png')
            plt.close()
        else:
            time_window_in_seconds = [t / 1000 for t in in_patch_time_window]
            plt.figure()
            plt.subplot(2,1,1)
            foraging_line, = plt.plot(time_window_in_seconds, forage_state, color = forage_colour, label = 'foraging poke')
            reward_line, = plt.plot(time_window_in_seconds, reward_state, 'orange', label = 'reward available')
            plt.title('patch number ' + str(p+1))
            plt.xlabel('Time (s)')
            plt.yticks([-1, 0, 1], labels = ['reward available', 'disengaged', 'in forage poke'])
            
            plt.savefig('mouse %s/session %s/timeline/patch %s.png' %(subject, date,  patch['patch number']), format = 'png')
            plt.close()    

# use SummaryMeasures instead
'''def AnalysePatch(patch):
    dwell_time = patch['end'] - patch['start']
    
    n_rewards = len(patch['forage times'])
    n_forage_pokes_per_reward = [len(trial) for trial in patch['forage times']]
    
    bout_durations = [[bout[1] - bout[0] for bout in trial] for trial in patch['forage times']]
    forage_per_reward = [sum(durations) for durations in bout_durations]
    successful_forage_duration = sum(forage_per_reward)
    
    n_pokes_before_giving_up = len(patch['forage before travel'])
    poke_durations_before_giving_up = [poke[1] - poke[0] for poke in patch['forage before travel']]
    give_up_time = sum(poke_durations_before_giving_up)
    
    n_forage_pokes = sum(n_forage_pokes_per_reward) + n_pokes_before_giving_up
    total_foraging = successful_forage_duration + give_up_time
    patch_engagement = total_foraging / dwell_time
    
    if patch['complete travel time']:
        travel_duration = patch['complete travel time'][1] - patch['complete travel time'][0]
        travel_poke_durations = [poke[1] - poke[0] for poke in patch['travel bouts']]
        n_travel_pokes = len(patch['travel bouts'])
        total_time_in_travel_poke = sum(travel_poke_durations)
        travel_engagement = total_time_in_travel_poke / travel_duration
    else:
        travel_duration = None
        travel_poke_durations = None
        n_travel_pokes = None
        total_time_in_travel_poke = None
        travel_engagement = None
    
    analysis = {'dwell time': dwell_time,
                'rewards': n_rewards,
                'pokes per reward': n_forage_pokes_per_reward,                   
                'successful forage bouts': bout_durations,
                'foraging per reward': forage_per_reward,
                'rewarded foraging': successful_forage_duration,
                'pokes before switching': n_pokes_before_giving_up,
                'duration of pokes before switching': poke_durations_before_giving_up,
                'total give up time': give_up_time,
                'total pokes': n_forage_pokes, 
                'total foraging': total_foraging,
                'patch engagement': patch_engagement,
                'duration of travel': travel_duration,
                'number of travel pokes': n_travel_pokes,
                'duration of travel pokes': travel_poke_durations,
                'total time in travel poke': total_time_in_travel_poke,
                'travel engagement':travel_engagement}
    
    return analysis '''

def SummaryMeasures(patches):
    n_patches = len(patches)
    
    dwell_times = [patch['end'] - patch['start'] for patch in patches]
    total_time_in_patches = sum(dwell_times)
    
    n_rewards = [len(patch['forage times']) for patch in patches]
    n_forage_pokes_per_reward = [[len(trial) for trial in patch['forage times']] for patch in patches]
    n_forage_pokes_per_patch = [sum(n) for n in n_forage_pokes_per_reward]
    
    forage_poke_durations = [[[bout[1] - bout [0] for bout in trial] for trial in patch['forage times']] for patch in patches]
    forage_poke_durations_per_reward = [[sum(durations) for durations in patch] for patch in forage_poke_durations]
    forage_poke_durations_per_patch = [sum(durations) for durations  in forage_poke_durations_per_reward]
    
    n_pokes_before_giving_up = [len(patch['forage before travel']) for patch in patches]
    poke_durations_before_giving_up = [[poke[1] - poke[0] for poke in patch['forage before travel']] for patch in patches]
    give_up_times = [sum(poke_durations) for poke_durations in poke_durations_before_giving_up]
    
    total_foraging = [sum(x) for x in zip(forage_poke_durations_per_patch, give_up_times)]
    total_n_forage_pokes = [s + g for s, g in zip(n_forage_pokes_per_patch,n_pokes_before_giving_up)]
    patch_engagement = [f / t for f, g, t in zip(total_foraging, give_up_times, dwell_times)]
    
    travel_durations = [patch['complete travel time'][1] - patch['complete travel time'][0]  for patch in patches if patch['complete travel time']]
    total_time_travelling = sum(travel_durations)
    
    n_travel_pokes = [len(patch['travel bouts']) for patch in patches]
    travel_poke_durations = [[poke[1] - poke[0] for poke in patch['travel bouts']] for patch in patches if patch['complete travel time']]
    total_time_in_travel_poke = [sum(durations) for durations in travel_poke_durations]
    
    travel_engagement = [p / t for p, t in zip(total_time_in_travel_poke, travel_durations)]
    
    if (sum(dwell_times) + sum(travel_durations)) > 0:
        overall_engagement = (sum(forage_poke_durations_per_patch) + sum(give_up_times) + sum(total_time_in_travel_poke)) / (sum(dwell_times) + sum(travel_durations))
    else:
        overall_engagement = []
    
    session_duration = total_time_in_patches + total_time_travelling
    
    description = {'number of patches': n_patches,
                   'dwell times': dwell_times, 
                   'total time in patches': total_time_in_patches,
                   'rewards per patch': n_rewards,
                   'number of pokes per reward': n_forage_pokes_per_reward,
                   'n_forage_pokes_per_patch': n_forage_pokes_per_patch,
                   'number of pokes per patch': total_n_forage_pokes,
                   'duration of forage pokes': [[[d for d in trial] for trial in patch] for patch in forage_poke_durations],
                   'forage time for each reward': [[d for d in reward] for reward in forage_poke_durations_per_reward],
                   'total successful forage time per patch': [d for d in forage_poke_durations_per_patch],
                   'number of pokes before switching': n_pokes_before_giving_up,
                   'duration of pokes before switching': [[d for d in patch] for patch in poke_durations_before_giving_up],
                   'give up time': give_up_times,
                   'total foraging per patch': total_foraging,
                   'patch engagement': patch_engagement,
                   'duration of travel': travel_durations,
                   'total travel time': total_time_travelling,
                   'number of travel pokes': n_travel_pokes,
                   'duration of travel pokes': [[d for d in patch] for patch in travel_poke_durations],
                   'total time in travel poke': total_time_in_travel_poke,
                   'travel engagement':travel_engagement,
                   'total duration': session_duration,
                   'overall engagement': overall_engagement
                           }
    
    return description

def BreakDownSession(file_path, unit = 's'):
    
    patches, subject, date = ExtractPatches(file_path, unit)
    
    # group different patches into categories of interest
    
    short_patches = [patch for patch in patches if patch['block'] == 'short']
    long_patches = [patch for patch in patches if patch['block'] == 'long']
    
    rich_patches = [patch for patch in patches if patch['richness'] == 'rich']
    medium_patches = [patch for patch in patches if patch['richness'] == 'medium']
    poor_patches = [patch for patch in patches if patch['richness'] == 'poor']
    
    short_and_rich_patches = [patch for patch in patches if (patch['block'] == 'short' and patch['richness'] == 'rich')]
    short_and_medium_patches = [patch for patch in patches if (patch['block'] == 'short' and patch['richness'] == 'medium')]
    short_and_poor_patches = [patch for patch in patches if (patch['block'] == 'short' and patch['richness'] == 'poor')]
    
    long_and_rich_patches = [patch for patch in patches if (patch['block'] == 'long' and patch['richness'] == 'rich')]
    long_and_medium_patches = [patch for patch in patches if (patch['block'] == 'long' and patch['richness'] == 'medium')]
    long_and_poor_patches = [patch for patch in patches if (patch['block'] == 'long' and patch['richness'] == 'poor')]
    
    categories = {'all': patches,
                  'short': short_patches, 'long': long_patches,
                  'rich': rich_patches, 'medium': medium_patches, 'poor': poor_patches,
                  'short x rich': short_and_rich_patches, 'short x medium': short_and_medium_patches, 'short x poor': short_and_poor_patches,
                  'long x rich': long_and_rich_patches, 'long x medium': long_and_medium_patches, 'long x poor': long_and_poor_patches}
    
    # get the different summaries for each category
    
    detailed_summary = {k: SummaryMeasures(v) for k, v in categories.items()}
    
    return detailed_summary

def FlattenMeasure(measure):
    
    flattened_measure = []
    for patch in measure:
        for element in patch:
            flattened_measure.append(element)
    
    return flattened_measure

def ConvertPerRewardMeasures(summary, measure, categories):
    n_rwds = max([len(patch) for patch in summary['all'][measure]])
    
    n_cats =len(categories)
    
    r = 1 / 3 #ratio of space between bars and bar width
    margin = 0.1 # extra space between bars of different rwd number
    width = (1- 2 * margin) / (n_cats * (r+1))
    off = r * width
 
    x = {cat:[i - 0.5 + margin + (off + width) / 2 + j * (width + off) for i in range(1, n_rwds+1)] for j, cat in enumerate(categories)}
    y = {cat:[[] for i in range(n_rwds)] for cat in categories}
    averages = {cat:[] for cat in categories}
    if measure == 'duration of forage pokes': # this measurement is repeated for each reward (because of bouts of foraging) and needs flattening
        for cat in categories:
            for i in range(n_rwds):
                for patch in summary[cat][measure]:
                    if len(patch) > i:
                        for bout in patch[i]:
                            y[cat][i].append(bout)
                averages[cat].append(np.mean(y[cat][i]))

    else:
        for cat in categories:
            for i in range(n_rwds):
                for patch in summary[cat][measure]:
                    if len(patch) > i:
                        y[cat][i].append(patch[i])
                averages[cat].append(np.mean(y[cat][i]))
        
    return x, width, y, averages

def PlotSessionSummary(file_path, unit = 's'):
    
    patches, subject, date = ExtractPatches(file_path, unit)
    
    summary = BreakDownSession(file_path, unit)
    
    list_of_single_measures = ['duration of travel',
                              'dwell times', 
                              'give up time',
                              'number of pokes before switching', 
                              'number of pokes per patch',
                              'number of travel pokes',
                              'patch engagement',
                              'rewards per patch',
                              'total successful forage time per patch',
                              'total time in travel poke',
                              'travel engagement']
    
    w = 0.5
    
    for measure in list_of_single_measures:
        
        
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
        x = [1,2]
        y = [np.mean(summary['short'][measure]), np.mean(summary['long'][measure])]
        ax1.bar(x, y, alpha = 0.2, width = w)
        dwell = [[d for d in summary['short'][measure]], [d for d in summary['long'][measure]]]
        for xe, ye in zip(x, dwell):
            ax1.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['short', 'long'])
    
         
        x = [1,2,3]
        y = [np.mean(summary['poor'][measure]), np.mean(summary['medium'][measure]), np.mean(summary['rich'][measure])]
        ax2.bar(x, y, alpha = 0.2, width = w)
        dwell = [[d for d in summary['poor'][measure]], [d for d in summary['medium'][measure]], [d for d in summary['rich'][measure]]]
        for xe, ye in zip(x, dwell):
            ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['poor', 'medium', 'rich'])
    
        w = 0.5
        x = [1, 2, 3, 5, 6, 7]
        y = [np.mean(summary['short x poor'][measure]), np.mean(summary['short x medium'][measure]), np.mean(summary['short x rich'][measure]),
              np.mean(summary['long x poor'][measure]), np.mean(summary['long x medium'][measure]), np.mean(summary['long x rich'][measure])]
        ax3.bar(x, y, alpha = 0.2, width = w)
        dwell = [[d for d in summary['short x poor'][measure]], [d for d in summary['short x medium'][measure]], [d for d in summary['short x rich'][measure]],
                  [d for d in summary['long x poor'][measure]], [d for d in summary['long x medium'][measure]], [d for d in summary['long x rich'][measure]]]
        for xe, ye in zip(x, dwell):
            #plt.scatter([xe] * len(ye), ye)
            ax3.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['short x poor', 'short x medium', 'short x rich',
                              'long x poor', 'long x medium', 'long x rich'])
        
        fig.suptitle(measure)
        
        plt.tight_layout()
        
        
        
        plt.savefig('mouse %s/session %s/summary/%s.png' %(subject, date,  measure), format = 'png')
        plt.close(fig)
    
    
    
    measures_per_reward = ['forage time for each reward',
                           'number of pokes per reward',
                           'duration of forage pokes']
    
    
        
    for measure in measures_per_reward:
        
        n_rwds = max([len(patch) for patch in summary['all'][measure]])    
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
            
        x, w, y, avg = ConvertPerRewardMeasures(summary, measure, ['short', 'long'])
       
        ax1.bar(x['short'], avg['short'], alpha = 0.2, width = w, label = 'short')
        ax1.bar(x['long'], avg['long'], alpha = 0.2, width = w, label = 'long')
        # pts = [short, long]
        for xe, ye in zip(x['short'], y['short']):
            ax1.scatter([xe] * len(ye), ye)
        for xe, ye in zip(x['long'], y['long']):
            ax1.scatter([xe] * len(ye), ye)
        ax1.set_xticks(range(1, n_rwds+1))
        ax1.set_xlabel('reward number')
        ax1.legend()
        
        
        x, w, y, avg = ConvertPerRewardMeasures(summary, measure, ['poor', 'medium', 'rich'])
        
        ax2.bar(x['poor'], avg['poor'], alpha =0.2, width = w, label = 'poor')
        ax2.bar(x['medium'], avg['medium'], alpha =0.2, width = w, label = 'medium')
        ax2.bar(x['rich'], avg['rich'], alpha =0.2, width = w, label = 'rich')
        for xe, ye in zip(x['poor'], y['poor']):
            ax2.scatter([xe] * len(ye), ye)
        for xe, ye in zip(x['medium'], y['medium']):
            ax2.scatter([xe] * len(ye), ye)
        for xe, ye in zip(x['rich'], y['rich']):
            ax2.scatter([xe] * len(ye), ye)
        ax2.set_xticks(range(1, n_rwds+1))
        ax2.set_xlabel('reward number')
        ax2.legend()
        
        
        x, w, y, avg = ConvertPerRewardMeasures(summary, measure, ['short x poor', 'long x poor',
                                                        'short x medium', 'long x medium',
                                                        'short x rich', 'long x rich'])
        
        ax3.bar(x['short x poor'], avg['short x poor'], alpha =0.2, width = w, label = 'short x poor')
        ax3.bar(x['short x medium'], avg['short x medium'], alpha =0.2, width = w, label = 'short x medium')
        ax3.bar(x['short x rich'], avg['short x rich'], alpha =0.2, width = w, label = 'short x rich')
        ax3.bar(x['long x poor'], avg['long x poor'], alpha =0.2, width = w, label = 'long x poor')
        ax3.bar(x['long x medium'], avg['long x medium'], alpha =0.2, width = w, label = 'long x medium')
        ax3.bar(x['long x rich'], avg['long x rich'], alpha =0.2, width = w, label = 'long x rich')
        ax3.set_xticks(range(1, n_rwds+1))
        ax3.set_xlabel('reward number')
        ax3.legend()
        
        fig.suptitle(measure)
        
        
        
        plt.savefig('mouse %s/session %s/summary/%s.png' %(subject, date,  measure), format = 'png')
        plt.close(fig)
        
    remaining_measures = ['duration of pokes before switching',
                          'duration of travel pokes']
    
    
    
    for measure in remaining_measures:
        
        short = FlattenMeasure(summary['short'][measure])
        long = FlattenMeasure(summary['long'][measure])
        
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
        x = [1,2]
        y = [np.mean(short), np.mean(long)]
        ax1.bar(x, y, alpha = 0.2, width = w)
        pts= [[p for p in short], [p for p in long]]
        for xe, ye in zip(x, pts):
            ax1.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['short', 'long'])
    
        
        poor = FlattenMeasure(summary['poor'][measure])
        medium = FlattenMeasure(summary['medium'][measure])
        rich = FlattenMeasure(summary['rich'][measure])    
        
        x = [1,2,3]
        y = [np.mean(poor), np.mean(medium), np.mean(rich)]
        ax2.bar(x, y, alpha = 0.2, width = w)
        pts = [[p for p in poor], [p for p in medium], [p for p in rich]]
        for xe, ye in zip(x, pts):
            ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['poor', 'medium', 'rich'])
    
    
        short_x_poor = FlattenMeasure(summary['short x poor'][measure])
        short_x_medium = FlattenMeasure(summary['short x medium'][measure])
        short_x_rich = FlattenMeasure(summary['short x rich'][measure])
        long_x_poor = FlattenMeasure(summary['long x poor'][measure])
        long_x_medium = FlattenMeasure(summary['long x medium'][measure])
        long_x_rich = FlattenMeasure(summary['long x rich'][measure])
        
        w = 0.5
        x = [1, 2, 3, 5, 6, 7]
        y = [np.mean(short_x_poor), np.mean(short_x_medium), np.mean(short_x_rich),
             np.mean(long_x_poor), np.mean(long_x_medium), np.mean(long_x_rich)]
        ax3.bar(x, y, alpha = 0.2, width = w)
        pts = [[p for p in short_x_poor], [p for p in short_x_medium], [p for p in short_x_rich],
               [p for p in long_x_poor], [p for p in long_x_medium], [p for p in long_x_rich]]
        for xe, ye in zip(x, pts):
            #plt.scatter([xe] * len(ye), ye)
            ax3.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['short x poor', 'short x medium', 'short x rich',
                              'long x poor', 'long x medium', 'long x rich'])
        
        fig.suptitle(measure)
        
        plt.tight_layout()
        
        plt.savefig('mouse %s/session %s/summary/%s.png' %(subject, date,  measure), format = 'png')
        plt.close(fig)

file_path = 'mouse 7/raw data/fp07-2019-03-05-104105.txt'

# session, subject, date = ExtractPatches(file_path, 's')
# des = SummaryMeasures([session[0]])
# analysis = AnalysePatch(session[0])
det = BreakDownSession(file_path, 's')
#PlotSessionTimeCourse(file_path)
#PlotPatchEvents(file_path)
PlotSessionSummary(file_path, 's')
