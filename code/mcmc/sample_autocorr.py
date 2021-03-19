#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mhmcmc import autocorrelation
from numpy.random import default_rng


gen = default_rng(2021)
data = gen.normal(0, 1, size=(100))
k, corr = autocorrelation(data)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot()
ax.plot(k, corr, marker='.')
ax.set_xlabel('displacement: k')
ax.set_ylabel('autocorr for N(0,1)')
fig.tight_layout()
plt.savefig('sample_autocorr.png')
plt.show()
