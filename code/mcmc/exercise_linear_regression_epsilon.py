#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_linear_regression.csv')

def log_gamma(x,k=1e-3,t=1e3):
  return (k-1)*np.log(x)-x/t if x>0 else -1e10

def log_likelihood(x):
  if x[2] < 0: return -1e10
  delta = table.logM_B - (x[0] + x[1]*(table.logsig-np.log10(200)))
  sqsig = table.logM_B_err**2 + x[1]**2*table.logsig_err**2 + x[2]
  logpdf = -(delta**2/sqsig/2+np.log(sqsig)/2)
  return np.sum(logpdf)+log_gamma(1/x[2])


step = GaussianStep(np.array([0.02, 0.15, 0.03]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([8.0, 5.0, 0.5])
model.initialize(x0)

sample = model.generate(51000)
sample = sample[1000:]
sample[:,2] = np.sqrt(sample[:,2])

k, corr = autocorrelation(sample)
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(3,1,1)
ax1.plot(k, corr[:,0])
ax1.set_ylabel('autocorr: alpha')
ax1.set_xlim([-2000,2000])
ax2 = fig.add_subplot(3,1,2)
ax2.plot(k, corr[:,1])
ax2.set_ylabel('autocorr: beta')
ax2.set_xlim([-2000,2000])
ax3 = fig.add_subplot(3,1,3)
ax3.plot(k, corr[:,2])
ax3.set_ylabel('autocorr: epsilon')
ax3.set_xlabel('displacement: k')
ax3.set_xlim([-2000,2000])
fig.tight_layout()
fig.savefig('exercise_linear_regression_epsilon_autocorr.png')
display_trace(sample, output='exercise_linear_regression_epsilon_trace.png')


x = np.linspace(-0.5,0.5,50)
a,b,e = sample.mean(axis=0)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.fill_between(x, a+b*x-3*e, a+b*x+3*e, color='gray', alpha=0.05)
ax.fill_between(x, a+b*x-e, a+b*x+e, color='gray', alpha=0.10)
for _a,_b,_e in sample[::1000,:]:
  ax.plot(x, _a+_b*x, color='orange', alpha=0.05)
ax.errorbar(
  x = table.logsig-np.log10(200), y = table.logM_B,
  xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
ax.plot(x, a+b*x)
ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
fig.tight_layout()
fig.savefig('exercise_linear_regression_epsilon.png')
plt.show()

print(f'MCMC inference: alpha={a:.3f}, beta={b:.3f}, epsilon={e:.3f}')
