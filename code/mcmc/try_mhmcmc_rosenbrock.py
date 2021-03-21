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
sample = sample[1000:]

x = np.linspace(-6,10,1000)
y = np.linspace(-20,100,1000)
xx,yy = np.meshgrid(x,y)
xy = np.dstack((xx.flatten(),yy.flatten())).T
z = np.log(rosenbrock(xy).reshape(1000,1000))
lv = np.linspace(-2,4,13)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.contour(xx,yy,z,levels=lv, alpha=0.3)
ax.scatter(sample[::10,0], sample[::10,1], marker='.', s=1)
ax.set_xlabel('random variable: x0')
ax.set_ylabel('random variable: x1')
fig.tight_layout()
fig.savefig('try_mhmcmc_rosenbrock.png')

from mhmcmc import display_trace
display_trace(sample, output='try_mhmcmc_rosenbrock_trace.png')
