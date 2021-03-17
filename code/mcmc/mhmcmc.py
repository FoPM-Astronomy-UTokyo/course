#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy.random import default_rng
import numpy as np


class AbstractLikelihood:
  ''' A template of likelihood function.'''

  def eval(self, x: np.ndarray):
    ''' Evaluate the log-likelihood for the state `x`.

    Parameters:
      x (numpy.ndarray): An input state (1-dimensional vector).

    Returns:
      float: The log-likelihood, logL(x).
    '''
    pass


class AbstractProposalDistribution:
  ''' A template of proposed distribution. '''

  def draw(self, x0: np.ndarray):
    ''' Propose a new state.

    Parameters:
      x0 (numpy.ndarray): The current state.

    Returns:
      numpy.ndarray: A newly-proposed state.
    '''

  def eval(self, x: np.ndarray, x0: np.ndarray):
    ''' Evaluate the log-probability for the state `x`.

    Parameters:
      x  (numpy.ndarray): The proposed state.
      x0 (numpy.ndarray): The current state.

    Returns:
      float: The log-probability, logQ(x, x0).
    '''


class MHMCMCSampler:
  ''' MCMC sampler with the Metropolis-Hastings algorithm. '''

  def __init__(self, likelihood: AbstractLikelihood,
               proposal_dist: AbstractProposalDistribution, seed: int=2021):
    ''' Generate a MCMC sampler instatnce.

    Parameters:
      likelihood (obj): An instance to calculate log-likelihood.
        A sub-class of AbstractLikelihood is preferred.
      proposal_dist (obj): An instance to draw from a proposal distribution.
        A sub-class of AbstractProposalDistribution is preferred.
      seed (float, optional): Random seed value.
    '''
    self.likelihood = likelihood
    self.proposal_dist = proposal_dist
    self.random = default_rng(seed)
    self.initialize(None)

  def initialize(self, x0: np.ndarray):
    ''' Initialize state.

    Parameters:
      x0 (numpy.ndarray): An initial state (1-dimensional vector).
    '''
    self.state = x0

  def step(self):
    ''' Draw a next Monte-Carlo step.

    Returns:
      numpy.ndarray: The newly generated next state.
    '''
    proposal = self.proposal_dist.draw(self.state)
    log_alpha = \
      self.likelihood.eval(proposal) - self.likelihood.eval(self.state) \
      + self.proposal_dist.eval(self.state, proposal) \
      - self.proposal_dist.eval(proposal, self.state)
    log_u = np.log(self.random.uniform(0, 1))
    return proposal if (log_u < log_alpha) else self.state

  def generate(self, n_sample: int):
    ''' Generate N-samples

    Parameters:
      n_samples (int): Number of MCMC samples to generate.

    Returns:
      numpyn.ndarray: A table of generated MCMC samples.
    '''
    samples = []
    for n in range(n_sample):
      samples.append(self.state)
      self.state = self.step()
    return np.vstack(samples)


def display_results(res: np.ndarray, dim: object=None, skip: int=1,
                    output: str=None):
  ''' Display the trace and histogram of samples.

  Parameters:
    res (numpy.ndarray): MCMC sample with the size of [Nample, Ndim].
    dim (int, optional): Dimension to be investigated. If not specified,
        The first dimension is selected by default.
    skip (int, optional): Pick up every {skip} sample. If not specified,
        Do not skip any samples by default.
    output (str, optional): Save figure as image.
  '''
  import matplotlib.pyplot as plt
  if dim is None: dim = list(range(res.shape[1]))
  if not isinstance(dim,list): dim = [dim,]
  ndim = len(dim)
  fig = plt.figure()
  axes = []
  for n,d in enumerate(dim):
    axes.append([fig.add_subplot(ndim,2,1+n*2),
                 fig.add_subplot(ndim,2,2+n*2)])
  for ax,d in zip(axes,dim):
    val = res[::skip,d]
    ax[0].plot(val)
    ax[0].set_ylabel(f'trace of X{d}')
    ax[1].hist(val, bins=min(200,max(int(val.size/25),20)), density=True)
    ax[1].set_ylabel(f'frequency of X{d}')
  plt.tight_layout()
  if output: fig.savefig(output)
  plt.show()



if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap(description='demo for HMMCMCSampler')
  parser.add_argument(
    '-n', '--num', dest='num', metavar='n', type=int, default=10000,
    help='number of monte carlo samples.')
  parser.add_argument(
    '-b', '--burnin', dest='burn', metavar='n', type=int, default=1000,
    help='number of burn-in steps.')
  parser.add_argument(
    '--step', metavar='n', type=float, default=0.2,
    help='step width of a Monte Carlo step.')
  parser.add_argument(
    '--png', metavar='filename', type=str, help='output figure name.')

  args = parser.parse_args()

  class logL(AbstractLikelihood):
    def __init__(self, center=[3, 5, -1], sigma=1.0):
      self.center = np.array(center)
      self.sigma = sigma
    def eval(self, x):
      return np.sum(-(x-self.center)**2/2.0/self.sigma**2)

  class logQ(AbstractProposalDistribution):
    def __init__(self, step):
      self.step = step
      self.gen = default_rng(2021)
    def draw(self, x0):
      return x0 + self.gen.normal(0, self.step, size=x0.shape)
    def eval(self, x, x0):
      return np.sum(-(x-x0)**2/(2*self.step**2))

  mcmc = MHMCMCSampler(logL(), logQ(args.step))

  x0 = np.zeros(3)
  mcmc.initialize(x0)


  val = mcmc.generate(args.num + args.burn)
  val = val[args.burn:]

  display_results(val, skip=5, output=args.png)
