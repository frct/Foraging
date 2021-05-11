# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:07:48 2021

Reward rates for all mice for all sessions

@author: franc
"""

import matplotlib.pyplot as plt
import numpy as np
import import_pycontrol as ip
import analyse_behaviour as analysis

#dt = 100 # time step size in ms

# complete travel times analysis

# load data

# change to location of raw data (.txt files)
behaviour_data_folder = '../../raw_data/behaviour_data/'

experiment = ip.Experiment(behaviour_data_folder)

all_short_travel_reward_rate = []
all_long_travel_reward_rate = []

avg_short_travel_reward_rate = []
avg_long_travel_reward_rate = []

for mouse in experiment.subject_IDs:
    print("mouse number %s" %(mouse))
    mouse_data = experiment.get_sessions(mouse)

    n_sessions = len(mouse_data)
    
    short_travel_reward_rate = np.zeros((n_sessions))
    long_travel_reward_rate = np.zeros((n_sessions))
    
    for idx, session in enumerate(mouse_data):
        #print("session number %s" %(session))
        tt = analysis.time_to_complete_travel(session)[3]

        complete_travel_times = [t[1] * 1000 for t in tt]
        short_travel_times = [t[1] * 1000 for t in tt if t[0] == 1000]
        long_travel_times = [t[1] * 1000 for t in tt if t[0] == 4000]
        short_duration = sum(short_travel_times)
        long_duration = sum(long_travel_times)
        n_rewards_short_travel = 0
        n_rewards_long_travel = 0

        mean_travels = analysis.get_mean_travel(session)
        mean_travels.append(mean_travels[-1])


        for patch_number, patch in enumerate(session.patch_data):
            n_rewards = len(patch["forage_time"])
            stay_duration = np.nansum(np.append(patch["forage_time"], patch["give_up_time"]))
    
            if mean_travels[patch_number] == 1000:
                n_rewards_short_travel += n_rewards
                short_duration += stay_duration
            else:
                n_rewards_long_travel += n_rewards
                long_duration += stay_duration
                             
        short_travel_reward_rate[idx] = 1000 * n_rewards_short_travel / short_duration
        long_travel_reward_rate[idx] = 1000 * n_rewards_long_travel / long_duration
    all_short_travel_reward_rate.append(short_travel_reward_rate)
    all_long_travel_reward_rate.append(long_travel_reward_rate)
    
    avg_short_travel_reward_rate.append(np.mean(short_travel_reward_rate))
    avg_long_travel_reward_rate.append(np.mean(long_travel_reward_rate))

plt.figure()
plt.plot(np.array([avg_short_travel_reward_rate, avg_long_travel_reward_rate]))
plt.ylabel("Average reward rate")
plt.xticks([0,1], ["short travel", "long travel"])