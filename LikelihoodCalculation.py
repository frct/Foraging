# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:14:07 2021

@author: franc
"""

import RateComparisonModel as model
import import_pycontrol as ip
import numpy as np

dt = 100 # time step size in ms

#parameters optimised on fp10-2019-05-20-132138
alpha_env = 0.0003
alpha_patch = 0.003
beta = 6.1
offset = 1

this_session = ip.Session('../../raw_data/behaviour_data/fp10-2019-05-20-132138.txt')

[rho_env, delta_env, rho_patch, delta_patch, trial_likelihood, foraging_likelihood] = model.RewardRates(this_session, alpha_patch, alpha_env, offset, beta, dt)

TotalLogLikelihood = model.LogLikelihood([alpha_env, alpha_patch, offset, beta], this_session, dt)

ForagingLogLikelihood = -np.sum(np.log(foraging_likelihood))


#parameters optimised on fp01-2019-02-21-112604
alpha_env = 0.0001
alpha_patch = 0.0003
beta = 4.52
offset = 1

that_session = ip.Session('../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt')

[rho_env, delta_env, rho_patch, delta_patch, trial_likelihood, foraging_likelihood] = model.RewardRates(that_session, alpha_patch, alpha_env, offset, beta, dt)

TotalLogLikelihood2 = model.LogLikelihood([alpha_env, alpha_patch, offset, beta], that_session, dt)

ForagingLogLikelihood2 = -np.sum(np.log(foraging_likelihood))