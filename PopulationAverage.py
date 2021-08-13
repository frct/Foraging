# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:12:57 2021

@author: franc
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

mice = [1,2,3,7,8,10,11,12,13,14,16,17]
n_mice = len(mice)

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
                   'travel engagement',
                   'duration of pokes before switching',
                   'duration of travel pokes']

measures_per_reward = ['forage time for each reward',
                       'number of pokes per reward',
                       'duration of forage pokes']

w = 0.5

all_measures = single_measures + measures_per_reward

population_averages = {cat: {m : [] for m in all_measures} for cat in categories}

for mouse in mice:
    mouse_averages = pickle.load(open('mouse %s/average over all sessions/averages.p' %(mouse), 'rb'))
    
    for measure in all_measures:
        for cat in categories:
            population_averages[cat][measure].append(mouse_averages[cat][measure])

for measure in single_measures:
    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
        
    x = [1,2]
    y = [np.mean(population_averages['short'][measure]), np.mean(population_averages['long'][measure])]
    ax1.bar(x, y, alpha = 0.2, width = w)
    pts = [[pt for pt in population_averages['short'][measure]], [pt for pt in population_averages['long'][measure]]]
    for xe, ye in zip(x, pts):
        ax1.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['short', 'long'])
    
         
    x = [1,2,3]
    y = [np.mean(population_averages['poor'][measure]), np.mean(population_averages['medium'][measure]), np.mean(population_averages['rich'][measure])]
    ax2.bar(x, y, alpha = 0.2, width = w)
    pts = [[pt for pt in population_averages['poor'][measure]], [pt for pt in population_averages['medium'][measure]], [pt for pt in population_averages['rich'][measure]]]
    for xe, ye in zip(x, pts):
        ax2.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['poor', 'medium', 'rich'])
    
    x = [1, 2, 3, 5, 6, 7]
    y = [np.mean(population_averages['short x poor'][measure]), np.mean(population_averages['short x medium'][measure]), np.mean(population_averages['short x rich'][measure]),
         np.mean(population_averages['long x poor'][measure]), np.mean(population_averages['long x medium'][measure]), np.mean(population_averages['long x rich'][measure])]
    ax3.bar(x, y, alpha = 0.2, width = w)
    pts = [[pt for pt in population_averages['short x poor'][measure]], [pt for pt in population_averages['short x medium'][measure]], [pt for pt in population_averages['short x rich'][measure]],
           [pt for pt in population_averages['long x poor'][measure]], [pt for pt in population_averages['long x medium'][measure]], [pt for pt in population_averages['long x rich'][measure]]]
    for xe, ye in zip(x, pts):
        ax3.scatter(xe + np.random.random(len(ye)) * w - w / 2, ye)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['short x poor', 'short x medium', 'short x rich',
                              'long x poor', 'long x medium', 'long x rich'])
        
    fig.suptitle(measure)
        
    plt.tight_layout()
        
    plt.savefig('average mouse behaviour/%s.png' %(measure), format = 'png')
    plt.close(fig)

r = 1 / 3 #ratio of space between bars and bar width
margin = 0.1 # extra space between bars of different rwd number
    
for measure in measures_per_reward:
    
    n_rwds = max([max([len(population_averages[cat][measure][i]) for i in range(n_mice)]) for cat in categories] )
    
    # mice don't have the same number of max rewards per patch, so add padding to make lists the same size
    for mouse in range(n_mice):
        for cat in categories:
            nan_list = [np.nan] * (n_rwds - len(population_averages[cat][measure][mouse]))
            population_averages[cat][measure][mouse].extend(nan_list)
    
    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (10,15))
            
    width = (1- 2 * margin) / (2 * (r+1))
    off = r * width
    x = {cat:[i - 0.5 + margin + (off + width) / 2 + j * (width + off) for i in range(1, n_rwds+1)] for j, cat in enumerate(['short', 'long'])}
    y = [np.nanmean([population_averages['short'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax1.bar(x['short'], y, alpha = 0.2, width = width, label = 'short')
    
    y = [np.nanmean([population_averages['long'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax1.bar(x['long'], y, alpha = 0.2, width = width, label = 'long')
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
    y = [np.nanmean([population_averages['poor'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax2.bar(x['poor'], y, alpha =0.2, width = width, label = 'poor')
    
    y = [np.nanmean([population_averages['medium'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax2.bar(x['medium'], y, alpha =0.2, width = width, label = 'medium')
    
    y = [np.nanmean([population_averages['rich'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax2.bar(x['rich'], y, alpha =0.2, width = width, label = 'rich')
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
    
    y = [np.nanmean([population_averages['short x poor'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax3.bar(x['short x poor'], y, alpha =0.2, width = width, label = 'short x poor')
    
    y = [np.nanmean([population_averages['short x medium'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax3.bar(x['short x medium'], y, alpha =0.2, width = width, label = 'short x medium')
    
    y = [np.nanmean([population_averages['short x rich'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax3.bar(x['short x rich'], y, alpha =0.2, width = width, label = 'short x rich')
    
    y = [np.nanmean([population_averages['long x poor'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax3.bar(x['long x poor'], y, alpha =0.2, width = width, label = 'long x poor')
    
    y = [np.nanmean([population_averages['long x medium'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax3.bar(x['long x medium'], y, alpha =0.2, width = width, label = 'long x medium')
    
    y = [np.nanmean([population_averages['long x rich'][measure][m][r] for m in range(n_mice)]) for r in range(n_rwds)]
    ax3.bar(x['long x rich'], y, alpha =0.2, width = width, label = 'long x rich')
    ax3.set_xticks(range(1, n_rwds+1))
    ax3.set_xlabel('reward number')
    ax3.legend()
    
    fig.suptitle(measure)
    plt.tight_layout()
    plt.savefig('average mouse behaviour/%s.png' %(measure), format = 'png')
    #plt.close(fig)