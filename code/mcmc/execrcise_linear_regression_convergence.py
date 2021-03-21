#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation

table = pd.read_csv('../../data/mcmc/exercise_linear_regression.csv')

def log_likelihood(x):
  delta = table.logM_B - (x[0] + x[1]*(table.logsig-np.log10(200)))
  sigma = table.logM_B_err
  return -np.sum(delta**2/sigma**2/2)

step = GaussianStep(np.array([0.02, 0.15]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([7.0, 8.0])
model.initialize(x0)

sample = model.generate(1000)

x = np.linspace(-0.5,0.5,50)
a,b = sample[400:].mean(axis=0)

fig = plt.figure(figsize=(8,6))

def update(i, sample):
  fig.clf()
  ax = fig.add_subplot()
  ax.errorbar(
    x = table.logsig-np.log10(200), y = table.logM_B,
    xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
  for n in range(10):
    _a,_b = sample[10*i+n,:]
    ax.plot(x, _a+_b*x, color='orange', alpha=0.1)
  ax.plot(x, a+b*x)

  ax.text(-0.48, 10.78, f'epoch: {10*i+1:3d}-{10*i+10}', fontsize=12)
  ax.set_xlim([-0.5,0.5])
  ax.set_ylim([5.5,11.0])
  ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
  ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
  fig.tight_layout()

import matplotlib.animation as animation
ani = animation.FuncAnimation(
  fig, update, fargs = (sample,), interval=100, frames=60)
ani.save('execrcise_linear_regression_convergence.gif', writer='imagemagick')
