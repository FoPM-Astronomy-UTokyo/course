#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


lam = 3

def log_likelihood(x):
  return -lam*x if x > 0 else -9999


step = GaussianStep(0.5)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.ones(1)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

x = np.linspace(0,5,1000)
f = lambda x: np.exp(-lam*x)*lam

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(sample, bins=50, density=True)
ax.plot(x, f(x))
ax.set_xlabel('random variable: x')
ax.set_ylabel('frequency')
fig.tight_layout()
plt.savefig('try_mhmcmc_exponential.png')
plt.show()
