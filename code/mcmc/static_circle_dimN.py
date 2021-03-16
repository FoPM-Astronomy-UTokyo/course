#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.special import gamma
import numpy as np
gen = default_rng(2021)

func = lambda x: ((x*x).sum(axis=1) < 1.0)

D = 15
A = 2**D
N = 1000000
n = np.arange(N)+1.0
x = gen.uniform(-1,1,size=(N,D))
w = A*func(x)
V = np.pi**(D/2)/gamma(D/2+1)

print(f'estimated area: {w.sum()/N:.5f} ({V:.5f})')

fig = plt.figure()
ax = fig.add_subplot()
ax.semilogx(n, w.cumsum()/n)
ax.semilogx(n, V*np.ones_like(n))
ax.set_ylabel('estimated volume')
ax.set_xlabel('number of samples')
plt.tight_layout()
fig.savefig('./static_circle_dimN.png')
plt.show()
