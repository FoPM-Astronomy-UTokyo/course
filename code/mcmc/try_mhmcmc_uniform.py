#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


lb,ub = -2, 3

def log_likelihood(x):
  return 1 if (lb < x < ub) else -9999


step = GaussianStep(0.5)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(1)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

x = np.linspace(-5,5,1000)
f = lambda x: ((lb<x)&(x<ub))/(ub-lb)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(sample, bins=50, density=True)
ax.plot(x, f(x))
ax.set_xlabel('random variable: x')
ax.set_ylabel('frequency')
fig.tight_layout()
fig.savefig('try_mhmcmc_uniform.png')
plt.show()
