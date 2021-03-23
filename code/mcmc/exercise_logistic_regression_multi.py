#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import display_trace, autocorrelation


table = pd.read_csv('../../data/mcmc/exercise_logistic_regression.csv')
dat = table.loc[:,['redshift','logab','fracDeV']].to_numpy()


def logistic(z):
  return 1/(1+np.exp(-z))

def log_likelihood(x):
  p = logistic(x[0] + np.dot(dat,x[1:].reshape((-1,1))).flatten())
  y = table.redspiral
  return np.sum(np.log(np.clip(y*p+(1-y)*(1-p),1e-30,1)))-np.sum(x**2)/2


step = GaussianStep(np.array([0.3, 0.9, 0.5, 0.5]))
model = MHMCMCSampler(log_likelihood, step)
x0 = np.array([-2.0, 0.0, 0.0, 0.0])
model.initialize(x0)

sample = model.generate(51000)
sample = sample[1000:]

k, corr = autocorrelation(sample)
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(4,1,1)
ax1.plot(k, corr[:,0])
ax1.set_ylabel('autocorr: b0')
ax1.set_xlim([-2000,2000])
ax2 = fig.add_subplot(4,1,2)
ax2.plot(k, corr[:,1])
ax2.set_ylabel('autocorr: b1')
ax2.set_xlim([-2000,2000])
ax3 = fig.add_subplot(4,1,3)
ax3.plot(k, corr[:,2])
ax3.set_ylabel('autocorr: b2')
ax3.set_xlim([-2000,2000])
ax4 = fig.add_subplot(4,1,4)
ax4.plot(k, corr[:,3])
ax4.set_ylabel('autocorr: b3')
ax4.set_xlim([-2000,2000])
ax4.set_xlabel('displacement: k')
fig.tight_layout()
fig.savefig('exercise_logistic_regression_multi_autocorr.png')
display_trace(sample, output='exercise_logistic_regression_multi_trace.png')


beta = sample.mean(axis=0)
R,B = (table.redspiral==1),(table.redspiral!=1)
gen = default_rng(2021)

rs = table.redshift
ab = table.logab+gen.uniform(0,0.01,size=table.shape[0])
fD = table.fracDeV+gen.uniform(0,0.01,size=table.shape[0])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
lab_arr = np.linspace(-0.02,0.22,300)
fdV_arr = np.linspace(-0.02,0.52,300)
lab_mesh,fdV_mesh = np.meshgrid(lab_arr,fdV_arr)
grid = np.stack((lab_mesh.flatten(),fdV_mesh.flatten())).T
z_mesh = beta[0]+beta[1]*rs.mean()+np.dot(grid,beta[2:]).reshape((300,300))
p_mesh = logistic(z_mesh)
ax.scatter(x=ab[B], y=fD[B], s=1, c='royalblue', alpha=0.2)
ax.scatter(x=ab[R], y=fD[R], s=3, c='orangered', alpha=0.5)
levels = np.linspace(0.02,0.29,10)
cb = ax.contour(lab_arr,fdV_arr,p_mesh,levels=levels)
ax.clabel(cb, inline=True)
ax.set_xlabel('$\log(a/b)$')
ax.set_ylabel('frac DeV')
fig.tight_layout()
plt.savefig('exercise_logistic_regression_multi.png')
plt.show()

print(f'MCMC inference: beta0={beta[0]: .3f} (const)')
print(f'              : beta1={beta[1]: .3f} (redshift)')
print(f'              : beta3={beta[2]: .3f} (logab)')
print(f'              : beta4={beta[3]: .3f} (fracDeV)')
