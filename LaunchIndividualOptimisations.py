
"""
Created on Mon Aug 23 12:00:09 2021

Instead of optimising sessions separately, optimise model by individual

@author: franc
"""

import ModelOptimisation as opt
import os
import ProcessRawData as ssa
import pickle



unit = 'ms'
dt = 100 # time step size in ms
Focused = True # time
model = 'Double bias'

mice = [7,8,10,11,12,13,14,16,17]
n_mice = len(mice)


for mouse in mice:
    print("mouse number %s" %(mouse))
    
    # get mouse sessions
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    files = os.listdir(folder_path)
    
    prepared_sessions = []
    
    for session_idx, file in enumerate(files):
        filepath = folder_path + '/' + file
        this_session, subject, date = ssa.ExtractPatches(filepath, unit)
        model_vars = opt.PrepareForOptimisation(this_session, dt, Focused)
        prepared_sessions.append(model_vars)  
    
    optimisation_results = opt.Optimise(prepared_sessions, model)

    pickle.dump(optimisation_results, open('Model optimisations/' + model + '/optimisation results for mouse ' + str(mouse) + '.p', 'wb'))