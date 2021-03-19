#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))

nchain = 4
step = GaussianStep(5.0)
samples = []
for n in range(nchain):
  x0 = np.zeros(2)
  model = MHMCMCSampler(log_likelihood, step)
  model.initialize(x0)
  sample = model.generate(101000)
  samples.append(sample[1000::])

samples = np.stack(samples)

nsample = samples.shape[1]
intra_mean  = samples.mean(axis=1)
global_mean = samples.mean(axis=(0,1)).reshape((1,2))
within_var  = samples.var(axis=1).mean(axis=0)
between_var = nsample*np.var(intra_mean-global_mean,axis=0)
Rhat = np.sqrt(1 + (between_var/within_var-1)/nsample)
print(f'Rhat values: {Rhat}')
