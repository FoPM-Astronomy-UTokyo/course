#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

u = gen.uniform(0,1,size=(3000))
v = gen.uniform(0,1,size=(3000))

r = np.sqrt(-2*np.log(1-u))
t = 2*np.pi*v
x = r*np.cos(t)
y = r*np.sin(t)

X = np.linspace(-5,5,500)
Y = np.exp(-X*X/2.0)/np.sqrt(2*np.pi)

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(x, bins=20, density=True)
ax.plot(X,Y)
ax.set_ylabel('frequency')
ax.set_xlabel('variable: x')
plt.tight_layout()
fig.savefig('./box-muller.png')
plt.show()
