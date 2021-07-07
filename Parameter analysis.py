# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:56:54 2021

plot individual parameter values

@author: franc
"""

import matplotlib.pyplot as plt
import pickle

optimised_mice = pickle.load(open( "saved_optimisations.p", "rb" ))

n_mice = len(optimised_mice)

alpha_patch = []
alpha_env = []
beta = []
reset = []
bias = []

n_sessions = []

for mouse in optimised_mice:
    n_sessions.append(len(optimised_mice[mouse]["convergence"]))
    alpha_env.append(optimised_mice[mouse]["parameters"][:,0])
    alpha_patch.append(optimised_mice[mouse]["parameters"][:,1])
    beta.append(optimised_mice[mouse]["parameters"][:,2])
    reset.append(optimised_mice[mouse]["parameters"][:,3])
    bias.append(optimised_mice[mouse]["parameters"][:,4])


plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], optimised_mice[mouse]["convergence"])
plt.xticks(range(n_mice))
plt.ylabel("convergence rate")
plt.xlabel("mice")

plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], optimised_mice[mouse]["n converged"])
plt.xticks(range(n_mice))
plt.ylabel("n converged optimisations")
plt.xlabel("mice")
  
plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], alpha_env[idx])
plt.xticks(range(n_mice))
plt.ylabel("alpha_env")
plt.xlabel("mice")

plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], alpha_patch[idx])
plt.xticks(range(n_mice))
plt.ylabel("alpha_patch")
plt.xlabel("mice")

plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], beta[idx])
plt.xticks(range(n_mice))
plt.ylabel("beta")
plt.xlabel("mice")

plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], reset[idx])
plt.xticks(range(n_mice))
plt.ylabel("reset")
plt.xlabel("mice")

plt.figure()
for idx, mouse in enumerate(optimised_mice):
    plt.scatter([idx] * n_sessions[idx], bias[idx])
plt.xticks(range(n_mice))
plt.ylabel("bias")
plt.xlabel("mice")