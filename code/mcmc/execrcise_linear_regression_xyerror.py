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
  sqsig = table.logM_B_err**2 + x[1]**2*table.logsig_err**2
  return -np.sum(delta**2/sqsig/2 - np.log(sqsig)/2.0)

step = GaussianStep(np.array([0.02, 0.15]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([8.0, 5.0])
model.initialize(x0)

sample = model.generate(51000)
sample = sample[1000:]

k, corr = autocorrelation(sample)
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(k, corr[:,0])
ax1.set_ylabel('autocorr: alpha')
ax1.set_xlim([-200,200])
ax2 = fig.add_subplot(2,1,2)
ax2.plot(k, corr[:,1])
ax2.set_ylabel('autocorr: beta')
ax2.set_xlabel('displacement: k')
ax2.set_xlim([-200,200])
fig.tight_layout()
fig.savefig('execrcise_linear_regression_xyerror_autocorr.png')
display_trace(sample, output='execrcise_linear_regression_xyerror_trace.png')


x = np.linspace(-0.5,0.5,50)
a,b = sample.mean(axis=0)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
for _a,_b in sample[::1000,:]:
  ax.plot(x, _a+_b*x, color='orange', alpha=0.1)
ax.errorbar(
  x = table.logsig-np.log10(200), y = table.logM_B,
  xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
ax.plot(x, a+b*x)
ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
fig.tight_layout()
fig.savefig('execrcise_linear_regression_xyerror.png')
plt.show()

print(f'MCMC inference: alpha={a:.3f}, beta={b:.3f}')
