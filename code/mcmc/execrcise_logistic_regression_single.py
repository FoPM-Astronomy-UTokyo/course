#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli as ber
from scipy.stats import norm as norm
from numpy.random import default_rng
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_logistic_regression.csv')

def logistic(z):
  return 1/(1+np.exp(-z))

def log_likelihood(x):
  p = logistic(x[0] + x[1]*table.fracDeV)
  y = table.redspiral
  return np.sum(np.log(y*p + (1-y)*(1-p)))


step = GaussianStep(np.array([0.50, 0.50]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([-5.0, 8.0])
model.initialize(x0)

sample = model.generate(51000)
sample = sample[1000:]

k, corr = autocorrelation(sample)
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(k, corr[:,0])
ax1.set_ylabel('autocorr: b0')
ax1.set_xlim([-2000,2000])
ax2 = fig.add_subplot(2,1,2)
ax2.plot(k, corr[:,1])
ax2.set_ylabel('autocorr: b1')
ax2.set_xlim([-2000,2000])
ax2.set_xlabel('displacement: k')
fig.tight_layout()
fig.savefig('execrcise_logistic_regression_single_autocorr.png')
display_trace(sample, output='execrcise_logistic_regression_single_trace.png')


beta = sample.mean(axis=0)
fdev = np.linspace(0,0.5,200)
parr = logistic(
  sample[:,0].reshape((1,-1))
  + sample[:,1].reshape((1,-1))*(fdev.reshape((200,1))))
psig = lambda x: np.percentile(parr, 100*norm.cdf(x), axis=1)
p   = parr.mean(axis=1)
R,B = (table.redspiral==1),(table.redspiral!=1)
gen = default_rng(2021)
scx = gen.uniform(0.,0.01,size=table.shape[0])
scy = gen.uniform(0.,0.10,size=table.shape[0])
sig = p*(1-p)

bins = np.linspace(0,0.5,20)
nr,_ = np.histogram(table.fracDeV[R], bins=bins)
nb,_ = np.histogram(table.fracDeV[B], bins=bins)
frac = nr/(nr+nb)
efrac = np.sqrt(nr)/(nr+nb)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.fill_between(fdev, psig(-3), psig(3), color='gray', alpha=0.05)
ax.fill_between(fdev, psig(-1), psig(1), color='gray', alpha=0.10)
ax.errorbar(
  bins[:-1], frac, yerr=efrac, fmt='.', c='forestgreen')
ax.scatter(
  x = (table.fracDeV+scx)[R], y = (table.redspiral+scy)[R],
  s=3, c='orangered')
ax.scatter(
  x = (table.fracDeV+scx)[B], y = (table.redspiral-scy)[B],
  s=1, alpha=0.2, c='royalblue')
ax.plot(fdev, p, c='orange')
ax.set_xlabel('frac DeV')
ax.set_ylabel('prob (red spiral)')
fig.tight_layout()
plt.savefig('execrcise_logistic_regression_single.png')
plt.show()

print(f'MCMC inference: beta0={beta[0]: .3f} (const)')
print(f'              : beta1={beta[1]: .3f} (fracDeV)')
