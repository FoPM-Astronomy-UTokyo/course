#!/usr/bin/env python
# -*- coding: utf-8 -*-
from jax.config import config
config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC


filenames = [
  './data/excercise_1_sequence_1.txt',
  './data/excercise_1_sequence_2.txt',
  './data/excercise_1_sequence_3.txt',
  './data/excercise_1_sequence_4.txt',
  './data/excercise_1_sequence_5.txt',
]

datasets = []
for fn in filenames:
  datasets.append(np.loadtxt(fn))


def noise_kernel(X, Z, noise):
  nx,nz = X.size,Z.size
  return noise*jnp.eye(nx,nz)

def rbf_kernel(X, Z, var, length):
  dXsq = jnp.power((X[:,None]-Z)/length,2)
  return var*jnp.exp(-0.5*dXsq)

def model(X, Y):
  var    = numpyro.sample('variance', dist.LogNormal(0.0, 1.0))
  length = numpyro.sample('lengthscale', dist.LogNormal(0.0, 1.0))
  noise  = numpyro.sample('noise', dist.LogNormal(0.0, 1.0))

  K = rbf_kernel(X,X,var,length) + noise_kernel(X,X,noise)

  numpyro.sample(
    'y',
    dist.MultivariateNormal(
      loc=jnp.zeros(X.shape[0]),
      covariance_matrix=K
    ),
    obs=Y,
  )

sampler = NUTS(model)
options = {
  'num_warmup':   500,
  'num_samples':  500,
  'num_chains':   1,
  'progress_bar': True,
}
mcmc = MCMC(sampler, **options)
keys = random.split(random.PRNGKey(42),5)

samples = []
for n,d in enumerate(datasets):
  print(f'Sequence {n+1}')
  X = jnp.array(d[:,0])
  Y = jnp.array(d[:,1])
  mcmc.run(keys[n],X,Y)
  mcmc.print_summary()
  samples.append(mcmc.get_samples())

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(12,9))
for n,s in enumerate(samples):
    ax.scatter(
        s['lengthscale'], s['variance'], 15,
        alpha=0.3, label=f'sample {n+1}')
ax.legend(fontsize=16)
ax.set_xlabel('Lengthscale', fontsize=16)
ax.set_ylabel('Variance', fontsize=16)
plt.show()