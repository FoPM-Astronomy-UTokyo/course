#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_count_model_regression.csv')
print(table)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.errorbar(
  x = table.M_K, y = table.N_gc,
  xerr = table.M_K_err, yerr = table.N_gc_err, fmt='.')
ax.set_xlabel('K-band magnitude: $M_K$')
ax.set_ylabel('Number of Globular Clusters')
ax.set_xlim([-19.5,-27.5])
ax.set_ylim([1,5e4])
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('exercise_count_model_regression_quick_view.png')
plt.show()
