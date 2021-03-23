#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng


table = pd.read_csv('../../data/mcmc/exercise_logistic_regression.csv')
print(table)

gen = default_rng(2021)
cl = table.redspiral+gen.normal(0.0,0.05,size=len(table))
rs = table.redshift+gen.uniform(0.00,0.01,size=len(table))
ab = table.logab+gen.uniform(0.00,0.01,size=len(table))
fD = table.fracDeV+gen.uniform(0.00,0.01,size=len(table))
R,B = table.redspiral==1, table.redspiral!=1
sca = gen.uniform(0,0.01,size=len(table))
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(3,1,1)
ax1.scatter(x = rs[B], y = cl[B], s=1, c='royalblue', alpha=0.5)
ax1.scatter(x = rs[R], y = cl[R], s=5, c='orangered', alpha=0.5)
ax1.set_xlabel('redshift')
ax1.set_yticks([0,1])
ax1.set_yticklabels(['blue','red'])
ax2 = fig.add_subplot(3,1,2)
ax2.scatter(x = ab[B], y = cl[B], s=1, c='royalblue', alpha=0.5)
ax2.scatter(x = ab[R], y = cl[R], s=5, c='orangered', alpha=0.5)
ax2.set_xlabel('$\log(a/b)$')
ax2.set_yticks([0,1])
ax2.set_yticklabels(['blue','red'])
ax3 = fig.add_subplot(3,1,3)
ax3.scatter(x = fD[B], y = cl[B], s=1, c='royalblue', alpha=0.5)
ax3.scatter(x = fD[R], y = cl[R], s=5, c='orangered', alpha=0.5)
ax3.set_xlabel('frac DeV')
ax3.set_yticks([0,1])
ax3.set_yticklabels(['blue','red'])
fig.tight_layout()
plt.savefig('exercise_logistic_regression_overview.png')
plt.show()
