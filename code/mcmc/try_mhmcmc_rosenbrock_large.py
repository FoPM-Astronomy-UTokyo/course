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


step = GaussianStep(100.)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

from mhmcmc import display_trace
display_trace(sample, output='try_mhmcmc_rosenbrock_trace_large.png')
