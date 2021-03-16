#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

func = lambda x: np.sqrt(np.clip(1-x*x,0,1))/np.pi*2.0

X = np.linspace(-1.2,1.2,500)
Y = func(X)

x = []
while len(x)<3000:
  u = gen.uniform(-2,2)
  v = gen.uniform(0,2.0/np.pi)
  if v<func(u):
    x.append(u)
x = np.array(x)

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(x, bins=20, density=True)
ax.plot(X,Y)
ax.set_ylabel('frequency')
ax.set_xlabel('variable: x')
plt.tight_layout()
fig.savefig('./reject.png')
plt.show()
