#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_linear_regression.csv')
print(table)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.errorbar(
  x = table.logsig, y = table.logM_B,
  xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
fig.tight_layout()
fig.savefig('execrcise_linear_regression_quick_view.png')
plt.show()
