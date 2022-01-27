# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:55:52 2021

@author: franc
"""

import os
import ModelOptimisation as opt
import pickle
import SingleSessionAnalysis as ssa
import BehaviourModels as mdl
import pandas as pd
import numpy as np
from scipy.stats import chi2

def LikelihoodRatioTest(LL_mdl, LL_0, df=5):
    D = [2 * (l0 - l1) for l0, l1 in zip(LL_0, LL_mdl)]
    p = [1 - chi2.cdf(d, df) for d in D]
    return p

def ParseSession(session, parameters, model):
    n_patches = len(session)
    model_vars = opt.PrepareForOptimisation(session, time_bin=0.1, Engaged=True)
   
    LL = opt.SessionLogLikelihood(parameters, model, model_vars)
        
    if model == 'Reward rate comparison':
        [rho_env, delta_env, rho_patch, delta_patch, p_action, p_leaving] = mdl.CompRwdRates_mdl(parameters,  **model_vars)
    elif model == 'Constant threshold':
        [threshold, rho_patch, delta_patch, p_action, p_leaving] = mdl.FixedThreshold_mdl(parameters,  **model_vars)
    else:
        raise Exception('model not recognised')
        
    n_departures = n_patches
    n_foraging_bins = len(p_leaving)
    freq_leaving = n_departures / n_foraging_bins
        
    bias_null = np.log(1 - freq_leaving) - np.log(freq_leaving)
        
    if model == 'Reward rate comparison':
        null_parameters = np.array([0,0,0,0,bias_null])
    elif model == 'Constant threshold':
        null_parameters = np.array([0,0,0,0,0,bias_null])
        
    LL0 = opt.SessionLogLikelihood(null_parameters, model, model_vars)
    return LL, LL0, bias_null, n_foraging_bins, rho_env, rho_patch, delta_patch, delta_env, p_action, p_leaving


def ParseMouse(mouse, parameters, model, SeparateSessions=True, SaveToExcel=False):
    folder_path = 'mouse ' + str(mouse) + '/raw data'
    sessions = os.listdir(folder_path)
    
    if SeparateSessions:
        n_sessions, n_parameters = np.shape(parameters)
    else:
        n_parameters = len(parameters)
        n_sessions = len(sessions)
    
    LogL = []
    LogL0 = []
    bias_null = []
    n_trials = []
    
    for ses_id, session in enumerate(sessions):
        if SeparateSessions:
            ses_parameters = parameters[ses_id,:]
        else:
            ses_parameters = parameters
        
        this_session = ssa.ExtractPatches(folder_path + '/' + session, unit= 's')[0]
        
        ll, ll0, b, nt, rho_env, rho_patch, delta_patch, delta_env, p_action, p_leaving = ParseSession(this_session, ses_parameters, model)
        
        LogL.append(ll)
        LogL0.append(ll0)
        bias_null.append(b)
        n_trials.append(nt)
        
        simulated_variables = {'env rwd rate': rho_env,
                               'patch rwd rate': rho_patch,
                               'env RPEs': delta_env,
                               'patch RPEs': delta_patch,
                               'instantaneous likelihood': p_action,
                               'prob leaving': p_leaving}
        
        pickle.dump(simulated_variables, open('Constrained simulations/' + model + '/Mouse ' + str(mouse) + '/session' + str(ses_id) + '.p', 'wb'))
        
    p = LikelihoodRatioTest(LogL, LogL0, df = n_parameters)
    
    AIC_0 = [2 * l for l in LogL0]
    AIC_model = [2 * n_parameters + 2 * l for l in LogL]
    
    BIC_0 = AIC_0
    BIC_model = [n_parameters * np.log(n) + 2 * l for n,l in zip(n_trials, LogL)]
    
    df = pd.DataFrame({'Session': [i for i in range(1, n_sessions + 1)],
                  'alpha_env': parameters[0],
                  'alpha_patch': parameters[1],
                  'beta': parameters[2],
                  'reset':  parameters[3],
                  'bias': parameters[4],
                  'LogL': LogL,
                  'null model bias': bias_null,
                  'LogL0': LogL0,
                  'LRT': p,
                  'AIC': AIC_model,
                  'AIC0': AIC_0,
                  'BIC': BIC_model,
                  'BIC0': BIC_0
                  })
    df.set_index('Session', inplace=True)
    
    if SaveToExcel:
        df.to_excel('mouse ' + str(mouse) + '/Session by session optimisation.xlsx')
        
    return df


ModelComparison = {}

mice = [1,2,3,7,8,10,11,12,13,14,16,17]
n_mice = len(mice)
Focused = True
model = 'Reward rate comparison'
SeparateSessions = False

#optimisation_results = pickle.load(open('Session by session optimisations.p', 'rb'))

for mouse in mice:
    if SeparateSessions: #load appropriate parameters
        pass
    else:     
        opt_res = pickle.load(open('Model optimisations/' +  model + '/Between sessions optimisations/optimisation results for mouse ' +str(mouse) + '.p', 'rb'))
    mouse_parameters = opt_res['parameters']
    df = ParseMouse(mouse, mouse_parameters, model, SeparateSessions, SaveToExcel=False)
    ModelComparison['mouse' + str(mouse)] = df
    
#pickle.dump(ModelComparison, open('Basic model evaluation.p', 'wb'))