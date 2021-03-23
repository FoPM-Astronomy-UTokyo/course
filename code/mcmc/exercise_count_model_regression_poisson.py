#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import pandas as pd
import matplotlib.pyplot as plt
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_count_model_regression.csv')

def log_likelihood(x):
  p = np.exp(x[0] + x[1]*table.M_K)
  return np.sum((table.N_gc)*np.log(p)-sp.gammaln(table.N_gc+1)-p)

step = GaussianStep(np.array([0.05, 0.002]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([-21.5, -1.2])
model.initialize(x0)

sample = model.generate(51000)
sample = sample[1000:]

k, corr = autocorrelation(sample)
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(k, corr[:,0])
ax1.set_ylabel('autocorr: alpha')
ax1.set_xlim([-1000,1000])
ax2 = fig.add_subplot(2,1,2)
ax2.plot(k, corr[:,1])
ax2.set_ylabel('autocorr: beta')
ax2.set_xlabel('displacement: k')
ax2.set_xlim([-1000,1000])
fig.tight_layout()
fig.savefig('exercise_count_model_regression_poisson_autocorr.png')
display_trace(
  sample, output='exercise_count_model_regression_poisson_trace.png')


M = np.linspace(-19.5,-27.5,50)
a,b = sample.mean(axis=0)
p,e = np.exp(a+b*M), np.sqrt(np.exp(a+b*M))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.fill_between(M, p-3*e, p+3*e, color='gray', alpha=0.05)
ax.fill_between(M, p-e, p+e, color='gray', alpha=0.10)
for _a,_b in sample[::100,:]:
  ax.plot(M, np.exp(_a+_b*M), color='orange', alpha=0.1)
ax.errorbar(
  x = table.M_K, y = table.N_gc,
  xerr = table.M_K_err, yerr = table.N_gc_err, fmt='.')
ax.plot(M, p)
ax.set_xlabel('K-band magnitude: $M_K$')
ax.set_ylabel('Number of Globular Clusters')
ax.set_xlim([-19.5,-27.5])
ax.set_ylim([1,5e4])
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('exercise_count_model_regression_poisson.png')
plt.show()

print(f'MCMC inference: alpha={a:.3f}, beta={b:.3f}')
