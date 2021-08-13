# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:11:07 2021

collapse session data by mouse to get mouse averages

@author: franc
"""

import os
import numpy as np
import SingleSessionAnalysis as ssa
from pathlib import Path
import matplotlib.pyplot as plt
import pickle


list_of_mice = [1,2,3,7,8,10,11,12,13,14,16,17]

categories = ['short', 'long',
                  'poor', 'medium', 'rich',
                  'short x poor', 'short x medium', 'short x rich',
                  'long x poor', 'long x medium', 'long x rich']

single_measures = ['duration of travel',
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

measures_per_reward = ['forage time for each reward',
                       'number of pokes per reward',
                       'duration of forage pokes']

remaining_measures = ['duration of pokes before switching',
                      'duration of travel pokes']

list_of_measures = single_measures + measures_per_reward + remaining_measures

for mouse in list_of_mice:
    
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    
    files = os.listdir(folder_path)
    
    
    session_averages = {cat: {m : [] for m in list_of_measures} for cat in categories}
    
    n_sessions = 0
        
    for file in files:
        n_sessions += 1
        
        filepath = folder_path + '/' + file
        
        patches, subject, date = ssa.ExtractPatches(filepath)
        
        summary = ssa.DescribeSession(patches, subject, date)
        
        
        
        for cat in categories:
            
            for measure in single_measures:
                session_averages[cat][measure].append(np.mean(summary[cat][measure]))
        
        
        
        for measure in measures_per_reward:
            x, w, y, avg = ssa.ConvertPerRewardMeasures(summary, measure, categories)
            for cat in categories:
                session_averages[cat][measure].append(avg[cat])
                
        
        
        for measure in remaining_measures:
            for cat in categories:
                flat = ssa.FlattenMeasure(summary[cat][measure])
                session_averages[cat][measure].append(np.mean(flat))
                
    mouse_averages = {cat: {m : np.mean(session_averages[cat][m]) for m in single_measures + remaining_measures} for cat in categories}
    
    # number of rewards differs between patches, so add a padding of nans to ensure all lists are the same length
    max_rwds = max([max([len(session_averages[cat]['forage time for each reward'][s]) for s in range(n_sessions)]) for cat in categories])
    
    for m in measures_per_reward:
        for cat in categories:
            for session in range(n_sessions):
                nan_list = [np.nan] * (max_rwds - len(session_averages[cat][m][session]))
                session_averages[cat][m][session].extend(nan_list)
    
    for cat in categories:
        for m in measures_per_reward:
            
            # re-order session averages so as to group measure values by reward number instead of session
            temp_list = [[session[r] for session in session_averages[cat][m]] for r in range(max_rwds)]
            
            mouse_averages[cat][m] = [np.nanmean(reward_nb) for reward_nb in temp_list]
            
    pickle.dump(mouse_averages, open('mouse %s/average over all sessions/averages.p' %(subject), 'wb'))
            
    for measure in single_measures + remaining_measures:
        
        
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
        
        x = [1,2]
        y = [mouse_averages['short'][measure], mouse_averages['long'][measure]]
        ax1.bar(x, y, alpha = 0.2, width = w)
        pts = [[pt for pt in session_averages['short'][measure]], [pt for pt in session_averages['long'][measure]]]
        for xe, ye in zip(x, pts):
            ax1.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['short', 'long'])
    
         
        x = [1,2,3]
        y = [mouse_averages['poor'][measure], mouse_averages['medium'][measure], mouse_averages['rich'][measure]]
        ax2.bar(x, y, alpha = 0.2, width = w)
        pts = [[pt for pt in session_averages['poor'][measure]], [pt for pt in session_averages['medium'][measure]], [pt for pt in session_averages['rich'][measure]]]
        for xe, ye in zip(x, pts):
            ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['poor', 'medium', 'rich'])
    
        w = 0.5
        x = [1, 2, 3, 5, 6, 7]
        y = [mouse_averages['short x poor'][measure], mouse_averages['short x medium'][measure], mouse_averages['short x rich'][measure],
             mouse_averages['long x poor'][measure], mouse_averages['long x medium'][measure], mouse_averages['long x rich'][measure]]
        ax3.bar(x, y, alpha = 0.2, width = w)
        pts = [[pt for pt in session_averages['short x poor'][measure]], [pt for pt in session_averages['short x medium'][measure]], [pt for pt in session_averages['short x rich'][measure]],
               [pt for pt in session_averages['long x poor'][measure]], [pt for pt in session_averages['long x medium'][measure]], [pt for pt in session_averages['long x rich'][measure]]]
        for xe, ye in zip(x, pts):
            #plt.scatter([xe] * len(ye), ye)
            ax3.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['short x poor', 'short x medium', 'short x rich',
                              'long x poor', 'long x medium', 'long x rich'])
        
        fig.suptitle(measure)
        
        plt.tight_layout()
        
        plt.savefig('mouse %s/average over all sessions/%s.png' %(subject, measure), format = 'png')
        plt.close(fig)
        
    r = 1 / 3 #ratio of space between bars and bar width
    margin = 0.1 # extra space between bars of different rwd number
    
    for measure in measures_per_reward:
        
        n_rwds = max([len(mouse_averages[cat][measure]) for cat in categories])    
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
            
        width = (1- 2 * margin) / (2 * (r+1))
        off = r * width
        x = {cat:[i - 0.5 + margin + (off + width) / 2 + j * (width + off) for i in range(1, n_rwds+1)] for j, cat in enumerate(['short', 'long'])}
       
        ax1.bar(x['short'], mouse_averages['short'][measure], alpha = 0.2, width = width, label = 'short')
        ax1.bar(x['long'], mouse_averages['long'][measure], alpha = 0.2, width = width, label = 'long')
        # pts = [short, long]
        # for xe, ye in zip(x['short'], y['short']):
        #     ax1.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        # for xe, ye in zip(x['long'], y['long']):
        #     ax1.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax1.set_xticks(range(1, n_rwds+1))
        ax1.set_xlabel('reward number')
        ax1.legend()
        
        
        width = (1- 2 * margin) / (3 * (r+1))
        off = r * width
        x = {cat:[i - 0.5 + margin + (off + width) / 2 + j * (width + off) for i in range(1, n_rwds+1)] for j, cat in enumerate(['poor', 'medium', 'rich'])}
        
        ax2.bar(x['poor'], mouse_averages['poor'][measure], alpha =0.2, width = width, label = 'poor')
        ax2.bar(x['medium'], mouse_averages['medium'][measure], alpha =0.2, width = width, label = 'medium')
        ax2.bar(x['rich'], mouse_averages['rich'][measure], alpha =0.2, width = width, label = 'rich')
        # for xe, ye in zip(x['poor'], y['poor']):
        #     ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        # for xe, ye in zip(x['medium'], y['medium']):
        #     ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        # for xe, ye in zip(x['rich'], y['rich']):
        #     ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
        ax2.set_xticks(range(1, n_rwds+1))
        ax2.set_xlabel('reward number')
        ax2.legend()
        
        width = (1- 2 * margin) / (3 * (r+1))
        off = r * width
        x = {cat:[i - 0.5 + margin + (off + width) / 2 + j * (width + off) for i in range(1, n_rwds+1)] for j, cat in enumerate(['short x poor', 'long x poor',
                                                                                                                                 'short x medium', 'long x medium',
                                                                                                                                 'short x rich', 'long x rich'])}
        
        ax3.bar(x['short x poor'], mouse_averages['short x poor'][measure], alpha =0.2, width = width, label = 'short x poor')
        ax3.bar(x['short x medium'], mouse_averages['short x medium'][measure], alpha =0.2, width = width, label = 'short x medium')
        ax3.bar(x['short x rich'], mouse_averages['short x rich'][measure], alpha =0.2, width = width, label = 'short x rich')
        ax3.bar(x['long x poor'], mouse_averages['long x poor'][measure], alpha =0.2, width = width, label = 'long x poor')
        ax3.bar(x['long x medium'], mouse_averages['long x medium'][measure], alpha =0.2, width = width, label = 'long x medium')
        ax3.bar(x['long x rich'], mouse_averages['long x rich'][measure], alpha =0.2, width = width, label = 'long x rich')
        ax3.set_xticks(range(1, n_rwds+1))
        ax3.set_xlabel('reward number')
        ax3.legend()
        
        fig.suptitle(measure)
        plt.tight_layout()
        plt.savefig('mouse %s/average over all sessions/%s.png' %(subject, measure), format = 'png')
        #plt.close(fig)