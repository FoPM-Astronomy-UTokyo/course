#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_count_model_regression.csv')

def log_likelihood(x):
  r = np.exp(x[2])
  m = np.exp(x[0] + x[1]*table.M_K)
  p = r/(r+m)
  G = sp.gammaln(r+table.N_gc)-sp.gammaln(r)-sp.gammaln(table.N_gc+1)
  return np.sum(G+table.N_gc*np.log(1-p)+r*np.log(p))

step = GaussianStep(np.array([0.15, 0.005, 0.05]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([-20, -1, 1])
model.initialize(x0)

sample = model.generate(51000)
sample = sample[1000:]

k, corr = autocorrelation(sample)
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(3,1,1)
ax1.plot(k, corr[:,0])
ax1.set_ylabel('autocorr: alpha')
ax1.set_xlim([-1000,1000])
ax2 = fig.add_subplot(3,1,2)
ax2.plot(k, corr[:,1])
ax2.set_ylabel('autocorr: beta')
ax2.set_xlim([-1000,1000])
ax3 = fig.add_subplot(3,1,3)
ax3.plot(k, corr[:,2])
ax3.set_ylabel('autocorr: gamma')
ax3.set_xlabel('displacement: k')
ax3.set_xlim([-1000,1000])
fig.tight_layout()
fig.savefig('exercise_count_model_regression_negbin_autocorr.png')
display_trace(
  sample, output='exercise_count_model_regression_negbin_trace.png')


M = np.linspace(-19.5,-27.5,50)
a,b,c = sample.mean(axis=0)
m,r = np.exp(a+b*M), np.exp(c)
p = r/(r+m)
e = np.sqrt(m/p)
rv = st.nbinom(r,p)
psig = lambda x: rv.ppf(st.norm.cdf(x))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.fill_between(M, psig(-3), psig(3), color='gray', alpha=0.05)
ax.fill_between(M, psig(-1), psig(1), color='gray', alpha=0.10)
for _a,_b,_e in sample[::1000,:]:
  ax.plot(M, np.exp(_a+_b*M), color='orange', alpha=0.1)
ax.errorbar(
  x = table.M_K, y = table.N_gc,
  xerr = table.M_K_err, yerr = table.N_gc_err, fmt='.')
ax.plot(M, m)
ax.set_xlabel('K-band magnitude: $M_K$')
ax.set_ylabel('Number of Globular Clusters')
ax.set_xlim([-19.5,-27.5])
ax.set_ylim([1,5e4])
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('exercise_count_model_regression_negbin.png')
plt.show()

print(f'MCMC inference: alpha={a:.3f}, beta={b:.3f}, gamma={c:.3f}')
