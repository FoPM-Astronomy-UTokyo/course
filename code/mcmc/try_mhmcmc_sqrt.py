#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


def log_likelihood(x):
  return np.log(np.sqrt(np.clip(1-x**2,1e-15,1)))


step = GaussianStep(0.5)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(1)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

x = np.linspace(-1.2,1.2,500)
f = lambda x: np.sqrt(np.clip(1-x**2,0,1))/np.pi*2.0

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(sample, bins=50, density=True)
ax.plot(x, f(x))
ax.set_xlabel('random variable: x')
ax.set_ylabel('frequency')
fig.tight_layout()
fig.savefig('try_mhmcmc_sqrt.png')
plt.show()
