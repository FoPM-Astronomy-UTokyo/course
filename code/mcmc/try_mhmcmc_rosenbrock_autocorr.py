#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import autocorrelation
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))

step = GaussianStep(5.0)
x0 = np.zeros(2)
model = MHMCMCSampler(log_likelihood, step)
model.initialize(x0)
sample = model.generate(101000)
sample = sample[1000:,:]

k, corr = autocorrelation(sample)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(k, corr[:,0], marker='')
ax1.set_xlim([-800,800])
ax1.set_ylabel('autocorr for x0')
ax2 = fig.add_subplot(2,1,2)
ax2.plot(k, corr[:,1], marker='')
ax2.set_xlim([-800,800])
ax2.set_ylabel('autocorr for x1')
ax2.set_xlabel('displacement: k')
fig.tight_layout()
fig.savefig('try_mhmcmc_rosenbrock_autocorr.png')
plt.show()
