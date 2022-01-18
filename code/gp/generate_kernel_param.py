#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from ast import parse
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist


def noise_kernel(X, Z, noise):
  nx,nz = X.size,Z.size
  return noise*jnp.eye(nx,nz)

def rbf_kernel(X, Z, var, length):
  dXsq = jnp.power((X[:,None]-Z)/length,2)
  return var*jnp.exp(-0.5*dXsq)

def matern12_kernel(X, Z, var, length):
  dX = jnp.abs((X[:, None] - Z) / length)
  return var * jnp.exp(-dX)


def generate_sequence(key, M, kernel, lb=0, ub=10):
  kx,ky = random.split(key,2)
  X = jnp.sort(dist.Uniform(lb,ub).expand([M,]).sample(kx))
  Y = dist.MultivariateNormal(
    loc=jnp.zeros(X.size),
    covariance_matrix=kernel(X,X)
  ).sample(ky)
  return jnp.stack((X,Y)).T

if __name__ == '__main__':
  parser = ap(
    description='generate sample data for kernel parameter estimation.')

  parser.add_argument(
    'M', type=int, help='number of measurements')
  parser.add_argument(
    'v', type=float, help='variance parameter')
  parser.add_argument(
    'l', type=float, help='lengthscale parameter')
  parser.add_argument(
    'output', type=str, help='output filename')
  parser.add_argument(
    '-n', '--noise', type=float, default=0.01,
    help='noise parameter')
  parser.add_argument(
    '-s', '--seed', type=int, default=42,
    help='random seed')

  args = parser.parse_args()
  key = random.PRNGKey(args.seed)

  kernel = lambda x,z: \
    rbf_kernel(x,z,var=args.v, length=args.l) \
    + noise_kernel(x,z,noise=args.noise)

  import numpy as np
  xy = generate_sequence(key, args.M, kernel)
  np.savetxt(args.output, xy)

  import matplotlib.pyplot as plt
  fig,ax = plt.subplots(figsize=(10,6))
  ax.plot(xy[:,0], xy[:,1])
  ax.set_xlabel('X', fontsize=16)
  ax.set_ylabel('Y', fontsize=16)
  plt.tight_layout()
  plt.show()