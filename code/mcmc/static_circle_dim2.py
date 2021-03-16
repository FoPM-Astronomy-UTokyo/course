#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

func = lambda x,y: (x**2+y**2 < 1.0)

A = 2*2
N = 100000
n = np.arange(N)+1.0
x = gen.uniform(-1,1,size=(N))
y = gen.uniform(-1,1,size=(N))
w = A*func(x,y)

print(f'estimated area: {w.sum()/N}')

fig = plt.figure()
ax = fig.add_subplot()
ax.semilogx(n, w.cumsum()/n)
ax.semilogx(n, np.pi*np.ones_like(n))
ax.set_ylabel('estimated area')
ax.set_xlabel('number of samples')
plt.tight_layout()
fig.savefig('./static_circle_dim2.png')
plt.show()
