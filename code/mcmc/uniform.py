#!/usr/bin/env python
import matplotlib.pyplot as plt

from numpy.random import default_rng
gen = default_rng(2021)
u = gen.uniform(0,1,size=(5))
print(u)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(gen.uniform(0,1,size=(1000)), ls='', marker='.')
ax.set_ylabel('uniform random value')
ax.set_xlabel('sample number')
plt.tight_layout()
fig.savefig('./uniform.png')
plt.show()
