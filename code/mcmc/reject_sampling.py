#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

lam = 3.0
ub  = 10.0
func = lambda x: lam*np.exp(-lam*x)/(1.0-np.exp(-lam*ub))

x = gen.uniform(0,ub, size=(10000))
a = gen.uniform(0,lam, size=(10000))
rejected = func(x) < a

print('rejected fraction: {}'.format(rejected.sum()/rejected.size))
