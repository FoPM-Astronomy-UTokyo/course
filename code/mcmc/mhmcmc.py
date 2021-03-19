#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Optional, Callable
from numpy.random import default_rng
from tqdm import tqdm, trange
import numpy as np


class AbstractProposalDistribution(object):
  ''' A template of proposal distribution class. '''

  def draw(self, x0: np.ndarray) -> np.ndarray:
    ''' Propose a new state.

    Parameters:
      x0 (numpy.ndarray): The current state.

    Returns:
      numpy.ndarray: A newly-proposed state.
    '''
    pass

  def eval(self, x: np.ndarray, x0: np.ndarray) -> float:
    ''' Evaluate the log-transition probability for the state `x`.

    Parameters:
      x  (numpy.ndarray): The proposed state.
      x0 (numpy.ndarray): The current state.

    Returns:
      float: The log-transition probability, logQ(x, x0).
    '''
    pass


class GaussianStep(AbstractProposalDistribution):
  ''' A random-walk proposal distribution with Gaussian distribution. '''

  def __init__(
      self, sigma: Union[float, np.ndarray], seed: int = 2021) -> None:
    ''' Generate an instance.

    Args:
      sigma (float or numpy.ndarray): Length of a Monte Carlo step.
      seed (int, optional): Seed value for the random value generator.
    '''
    self.sigma = sigma
    self.gen = default_rng(seed)

  def draw(self, x0: np.ndarray) -> np.ndarray:
    ''' Propose a new state by random walk.

    Parameters:
      x0 (numpy.ndarray): The current state.

    Returns:
      numpy.ndarray: A newly-proposed state.
    '''
    return x0 + self.gen.normal(0, self.sigma, size=x0.shape)

  def eval(self, x: np.ndarray, x0: np.ndarray) -> float:
    ''' Evaluate the log-transition probability for the state `x`.

    Parameters:
      x  (numpy.ndarray): The proposed state.
      x0 (numpy.ndarray): The current state.

    Returns:
      float: The log-transition probability, logQ(x, x0).
    '''
    return np.sum(-(x-x0)**2/(2*self.sigma**2))


class CauchyStep(AbstractProposalDistribution):
  ''' A random-walk proposal distribution with Cauchy distribution. '''

  def __init__(
      self, scale: Union[float, np.ndarray], seed: int = 2021) -> None:
    ''' Generate an instance.

    Args:
      scale (float or numpy.ndarray): Length of a Monte Carlo step.
      seed (int, optional): Seed value for the random value generator.
    '''
    self.scale = scale
    self.gen = default_rng(seed)

  def draw(self, x0: np.ndarray) -> np.ndarray:
    ''' Propose a new state by random walk.

    Parameters:
      x0 (numpy.ndarray): The current state.

    Returns:
      numpy.ndarray: A newly-proposed state.
    '''
    return x0 + self.scale * self.gen.standard_cauchy(size=x0.shape)

  def eval(self, x: np.ndarray, x0: np.ndarray) -> float:
    ''' Evaluate the log-transition probability for the state `x`.

    Parameters:
      x  (numpy.ndarray): The proposed state.
      x0 (numpy.ndarray): The current state.

    Returns:
      float: The log-transition probability, logQ(x, x0).
    '''
    return -np.sum(np.log(1.0 + ((x-x0)/self.scale)**2))


class MHMCMCSampler(object):
  ''' MCMC sampler with the Metropolis-Hastings algorithm. '''

  def __init__(self,
      log_likelihood: Callable[[np.ndarray], float],
      proposal_dist: AbstractProposalDistribution, seed: int=2021) -> None:
    ''' Generate a MCMC sampler instatnce.

    Parameters:
      likelihood (function): An instance to calculate log-likelihood.
        A sub-class of AbstractLikelihood is preferred.
      proposal_dist (AbstractProposalDistribution):
        An instance to draw from a proposal distribution.
        A sub-class of AbstractProposalDistribution is preferred.
      seed (float, optional): Random seed value.
    '''
    self.log_likelihood = log_likelihood
    self.proposal_dist = proposal_dist
    self.random = default_rng(seed)
    self.initialize(None)

  def initialize(self, x0: np.ndarray) -> None:
    ''' Initialize state.

    Parameters:
      x0 (numpy.ndarray): An initial state (1-dimensional vector).
    '''
    self.state = x0

  def step_forward(self) -> None:
    ''' Draw a next Monte-Carlo step.

    Returns:
      numpy.ndarray: The newly generated next state.
    '''
    proposal = self.proposal_dist.draw(self.state)
    log_alpha = \
      self.log_likelihood(proposal) - self.log_likelihood(self.state) \
      + self.proposal_dist.eval(self.state, proposal) \
      - self.proposal_dist.eval(proposal, self.state)
    log_u = np.log(self.random.uniform(0, 1))
    self.state = proposal if (log_u < log_alpha) else self.state

  def generate(self, n_sample: int) -> np.ndarray:
    ''' Generate N-samples

    Parameters:
      n_samples (int): Number of MCMC samples to generate.

    Returns:
      numpyn.ndarray: A table of generated MCMC samples.
    '''
    if self.state is None:
      raise RuntimeError('state is not initialized.')
    samples = []
    tqdmfmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
    for n in trange(n_sample, bar_format=tqdmfmt):
      self.step_forward()
      samples.append(self.state)
    return np.vstack(samples)


def display_results(
    res: np.ndarray,
    dim: Union[int, np.ndarray, None]=None,
    skip: int=1, output: str=None) -> None:
  ''' Display the trace and histogram of samples.

  Parameters:
    res (numpy.ndarray): MCMC sample with the size of [Nample, Ndim].
    dim (int or numpy.ndarray, optional):
        Dimension to be investigated. If not specified,
        The first dimension is selected by default.
    skip (int, optional): Pick up every {skip} sample. If not specified,
        Do not skip any samples by default.
    output (str, optional): Save figure as image.
  '''
  import matplotlib.pyplot as plt
  if dim is None: dim = list(range(res.shape[1]))
  if not isinstance(dim,list): dim = [dim,]
  ndim = len(dim)
  fig = plt.figure(figsize=(12,5))
  axes = []
  for n,d in enumerate(dim):
    axes.append([fig.add_subplot(ndim,2,1+n*2),
                 fig.add_subplot(ndim,2,2+n*2)])
  for ax,d in zip(axes,dim):
    val = res[::skip,d]
    ax[0].plot(val)
    ax[0].set_ylabel(f'trace of X{d}')
    ax[1].hist(val, bins=min(100,max(int(val.size/25),20)), density=True)
    ax[1].set_ylabel(f'frequency of X{d}')
  plt.tight_layout()
  if output: fig.savefig(output)
  plt.show()



if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap(description='demo for HMMCMCSampler')
  parser.add_argument(
    '-n', '--num', dest='num', metavar='n', type=int, default=3000,
    help='number of monte carlo samples.')
  parser.add_argument(
    '-b', '--burnin', dest='burn', metavar='n', type=int, default=1000,
    help='number of burn-in steps.')
  parser.add_argument(
    '--step', metavar='n', type=float, default=0.5,
    help='step width of a Monte Carlo step.')
  parser.add_argument(
    '--png', metavar='filename', type=str, help='output figure name.')

  args = parser.parse_args()

  def logL(x: np.array) -> float:
    center = np.array([3.2, 5.9, 1.1, 8.8])
    sigma  = np.array([0.8, 0.5, 0.4, 1.2])
    return np.sum([-((z-c)/s)**2/2 for z,c,s in zip(x,center,sigma)])

  step = GaussianStep(args.step)
  mcmc = MHMCMCSampler(logL, step)

  x0 = np.zeros(4)
  mcmc.initialize(x0)


  val = mcmc.generate(args.num + args.burn)
  val = val[args.burn:]

  mean = val.mean(axis=0)
  std  = val.std(axis=0)
  print('mean value')
  print('  ground truth   : [3.20 5.90 1.10 8.80]')
  print('  estimated value: [{:.2f} {:.2f} {:.2f} {:.2f}]'.format(*mean))
  print('stddev value')
  print('  ground truth   : [0.80 0.50 0.40 1.20]')
  print('  estimated value: [{:.2f} {:.2f} {:.2f} {:.2f}]'.format(*std))

  display_results(val, skip=30, output=args.png)
