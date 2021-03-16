#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

lam = 3.0
X = np.linspace(0,5,100)
Y = lam*np.exp(-lam*X)

u = gen.uniform(0,1,size=(10000))
x = -1.0/lam*np.log(1.0-u)

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(x, bins=50, density=True)
ax.plot(X,Y)
ax.set_ylabel('frequency')
ax.set_xlabel('variable: x')
plt.tight_layout()
fig.savefig('./exponential.png')
plt.show()
