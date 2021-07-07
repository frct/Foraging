# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:46:46 2021

analyse all single sessions of all mice

@author: franc
"""

import os
import SingleSessionAnalysis as ssa
from pathlib import Path

# filepath = 'mouse 12/raw data/fp12-2019-06-05-110506.txt'
# patches, subject, date = ssa.ExtractPatches(filepath)

list_of_mice = [14,16,17] #1,2,3,7,8,10,11,12,13,14,16,17]

for mouse in list_of_mice:
    
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    
    files = os.listdir(folder_path)
    
    for file in files:
        filepath = folder_path + '/' + file
        
        patches, subject, date = ssa.ExtractPatches(filepath)
        
        timeline_dir = 'mouse %s/session %s/timeline' %(subject, date)
        summary_dir = 'mouse %s/session %s/summary' %(subject, date)
        
        Path(timeline_dir).mkdir(parents=True, exist_ok=True)
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        
        ssa.PlotSessionTimeCourse(patches, subject, date, first_patch = 0, last_patch=None)
        ssa.PlotSessionSummary(patches, subject, date)
        ssa.PlotPatchEvents(patches, subject, date)