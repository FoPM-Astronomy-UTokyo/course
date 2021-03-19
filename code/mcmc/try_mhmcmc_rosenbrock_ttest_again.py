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


step = GaussianStep(5.0)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000::200]
m,n = sample.shape
m20,m50 = int(m/5), int(m/2)

from scipy import stats
tv,pv = stats.ttest_ind(
  sample[:m20,:], sample[m50:(m50+m20),:], equal_var=False)

print(f'number of individual sampling: {m}')
print(f'probability(x0[former] == x0[latter]): {pv[0]:g}')
print(f'probability(x1[former] == x1[latter]): {pv[1]:g}')
