#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


sigma = 0.2

def log_likelihood(x):
  return -(np.sqrt(x[0]**2 + x[1]**2) - 2)**2/2.0/sigma**2


step = GaussianStep(0.5)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(sample[::10,0], sample[::10,1], s=1, marker='.')
ax.set_xlabel('random variable: x0')
ax.set_ylabel('random variable: x1')
fig.tight_layout()
plt.savefig('try_mhmcmc_circle.png')
plt.show()
