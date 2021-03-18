#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


mu  = np.array([-1, 2])
cov = np.array([[5.0, 2.0], [2.0, 2.0]])

def log_likelihood(x):
  return -np.sum(np.dot(x-mu,np.linalg.solve(cov,x-mu))/2.0)


step = GaussianStep(0.5)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(sample[::20,0], sample[::20,1], marker='.')
ax.set_xlabel('random variable: x0')
ax.set_ylabel('random variable: x1')
fig.tight_layout()
plt.savefig('try_mhmcmc_normal.png')
plt.show()
