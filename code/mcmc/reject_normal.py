#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

func = lambda x: np.sqrt(np.clip(1-x*x,0,1))/np.pi*2.0
prop = lambda x: np.exp(-x*x/2.0)/np.sqrt(2.0*np.pi)

X = np.linspace(-1.2,1.2,500)
Y = func(X)

trial = 0
x = []
while len(x)<30000:
  u = gen.normal(0,1)
  v = gen.uniform(0,np.sqrt(8/np.pi))
  if v<func(u)/prop(u):
    x.append(u)
  trial += 1
x = np.array(x)
print(f'total trial: {trial}')

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(x, bins=50, density=True)
ax.plot(X,Y)
ax.set_ylabel('frequency')
ax.set_xlabel('variable: x')
plt.tight_layout()
fig.savefig('./reject_normal.png')
plt.show()
