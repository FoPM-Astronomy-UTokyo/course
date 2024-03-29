# MHMCMC チュートリアル

ここでは Metropolis-Hastings アルゴリズムを使ってマルコフ連鎖モンテカルロ計算をするためのプログラムを作成してみます. ただし, 実習時間内にすべてをスクラッチから作成するのは大変なので, 大まかな骨組みだけ Python のコードで用意しました. 核心となるパーツだけを Python で書くことで MCMC 計算を走らせることができます.[^1]

[^1]: ただし用意したコードはパフォーマンスについては何も考慮されていません. Python のパッケージとして使用可能な MCMC 計算ライブラリとは計算速度は雲泥の差があると思われます. ここでは, MCMC 計算の効率を体験するというよりは MCMC 計算のライブラリの中で何が起きているのかを意識しながらコードを書くことを目的としています.


## MHMCMC Sampler クラス

MCMC 計算をするためのライブラリとして [mhmcmc.py][mhmcmc] を用意しました. 各自ダウンロードして作業用ディレクトリに配置してください. ここでは以下の 2 つのクラスを使用します.

* {==MHMCMCSampler==}: Metropolis-Hastings アルゴリズムで MCMC 計算をするためのクラス.
* {==GaussianStep==}: 正規分布でランダムウォークすることで状態を提案するクラス.

それぞれのクラスの使い方の例は以下のとおりです.

``` python
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np

mu  = 5.0 # 分布の中心
sig = 0.7 # 分布の幅

# 乱数をサンプルする確率分布を定義
def log_likelihood(x):
  return -np.sum(((x-mu)**2)/2.0/sig**2)


# sigma = 0.5 でランダムウォークする提案分布を作成
step = GaussianStep(0.5)

# 確率分布と提案分布をあたえて MCMC 計算するモデルを作成
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(1)               # [0] を初期状態に指定
model.initialize(x0)           # 初期状態を設定
sample = model.generate(11000) # 11000 個のデータをサンプリング
sample = sample[1000:]         # 最初の 1000 個を捨てる (burn-in)

print('ground truth   : [{}, {}]'.format(mu, sig))
print('estimated value: [{}, {}]'.format(sample.mean(), sample.std()))
```

??? summary "計算結果"
    ```
    ground truth   : [5.000, 0.700]
    estimated value: [4.988, 0.702]
    ```

確率分布を与える `log_likelihood` は 1 次元の `numpy.ndarray()` を受け取って `float` を返す関数として定義してください. Metropolis-Hastings アルゴリズムでは確率分布の比さえ使えれば良いので, 確率分布は規格化されていなくても問題ありません.


### 計算の流れ

{==MHMCMCSampler==} では `generate()` という関数を呼ぶことで任意の個数のデータをサンプリングします. ここでは `step_forward()` という関数で現在の状態を更新した後に, `samples` に追加することを繰り返しています.

``` python
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
  for n in range(n_sample):
    self.step_forward()
    samples.append(self.state)
  return np.vstack(samples)
```

`step_forward()` の処理内容を以下に示しました. `proposal_dist` (提案分布を生成するクラス) から `draw()` という関数で次の状態への提案 `proposal` を生成します. 次に, 確率分布 (尤度関数) の対数を計算する `log_likelihood()` と遷移確率の対数を計算する関数 `proposal_dist.eval()` をつかって $\log\alpha$ を計算します. また, $[0,1)$ の一様分布からサンプルした値の対数 $\log{u}$ を計算します. $\log{u} < \log\alpha$ であれば提案された `proposal` が次の状態として採択されます. そうでない場合には現在の状態にとどまります. [前章](./markov_chain.md) で説明した Metropolis-Hastings アルゴリズムがそのまま実装されていることがわかると思います.

``` python
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
```


## MHMCMC によるサンプリング

ここでは簡単な形状の確率分布関数に従う乱数を生成してみます.


### 演習: 1 次元のケース

確率変数 $x$ が 1 次元のケースについて実際にサンプリングしてみます. 以下の形状を持つ以下の形状を持つ確率分布関数を定義して MCMC によってサンプリングしてください. また, 得られたヒストグラムを確率分布関数から期待される形状と比べてください.[^2]

[^2]: 得られたデータ列の分布を議論する場合には, サンプル間の相関が結果に影響します. 時間的に十分離れた場所のサンプルだけに間引いてから扱うことでサンプル間の相関を切ることができます. そのためにはより多くのデータをサンプリングする必要があるため, ここではこの操作は省略します.


__1. 確率分布が $\sqrt{1-x^2}$ に比例する乱数を生成してください.__

対数尤度関数を $\log\sqrt{1-x^2}$ に比例するように定義してください. ただし $1-x^2 < 0$ の場合には値が正しく定義されないので $10^{-15}$ で置き換えることにします.[^1.1] 確率分布は規格化されていなくても問題ありません.

[^1.1]: `numpy.clip(x, lower_bound, upper_bound)` という関数が使えます.

<details markdown=1><summary>Example</summary>
``` python
--8<-- "code/mcmc/try_mhmcmc_sqrt.py"
```
![生成されたデータのヒストグラム](img/try_mhmcmc_sqrt.png)
</details>


__2. 範囲 $[-2, 3)$ に一様に分布する乱数を生成してください.__

対数尤度関数として範囲内であれば 1 を, 範囲外であれば大きな負の値を返す関数を定義してください. 確率分布は規格化されていなくても問題ありません.

<details markdown=1><summary>Example</summary>
``` python
--8<-- "code/mcmc/try_mhmcmc_uniform.py"
```
![生成されたデータのヒストグラム](img/try_mhmcmc_uniform.png)
</details>


__3. スケールが $\lambda$ である指数分布に従う乱数を生成してください.__

指数分布は確率分布 $p(x) = \lambda\exp(-\lambda x)$ によって定義されます. 対数尤度は $-\lambda x$ に比例します. ただし $x < 0$ の場合には大きな負の値に置き換えてください. 規格化定数 $\log\lambda$ は対数尤度関数に含まなくても問題ありません.

<details markdown=1><summary>Example</summary>
``` python
--8<-- "code/mcmc/try_mhmcmc_exponential.py"
```
![生成されたデータのヒストグラム](img/try_mhmcmc_exponential.png)
</details>



### 演習: 2 次元のケース

確率変数 $x$ が 2 次元のケースについてサンプリングしてみます. 以下の形状を持つ以下の形状を持つ確率分布関数を定義して MCMC によってサンプリングしてください. また, 得られたデータで散布図を作成して期待通りのデータが得られていることを確認してください.


__1. $P(r) \propto \exp\left(-\frac{1}{2\sigma^2}(r-1)^2\right), ~~ r = \sqrt{x_{[0]}^2 + x_{[1]}^2}$ から乱数を生成してください.__

定義通りに対数尤度関数を定義してください. 確率分布は規格化されていなくても問題ありません.

<details markdown=1><summary>Example</summary>
``` python
--8<-- "code/mcmc/try_mhmcmc_circle.py"
```
![生成されたデータの散布図](img/try_mhmcmc_circle.png)
</details>


__2. 平均が (-1, 2), 分散が (5, 2), 共分散が 2 である 2 変数正規分布から乱数を生成してください.__

分散・共分散行列 $\Sigma$ と確率分布を以下のように定義すると $x$ の第 1 成分 $x_{[1]}$ の分散は 5, $x$ の第 2 成分 $x_{[2]}$ の分散は 2, $x_{[1]}$ と $x_{[2]}$ の共分散は 2 になります.

$$
\Sigma = \begin{pmatrix}5 & 2 \\ 2 & 2 \end{pmatrix},~~
p(x) \propto \exp\left(-\frac{1}{2}x^T\Sigma^{-1}x\right)
$$

$\Sigma$ の逆行列を求めて[^2.1]おけば比較的簡単に定義できると思います. あるいは $b = \Sigma^{-1}x$ とおくと, $b$ は $\Sigma$ を係数とする連立一次方程式 $x = \Sigma{b}$ の解になるという関係を使うこともできます.[^2.2] 以下の例では後者を採用しています. 確率分布は規格化されていなくても問題ありません.

[^2.1]: `numpy.linalg.inv(S)` という関数を使うことができます.
[^2.2]: 連立一次方程式の解は `numpy.linalg.solve(S,x)` で計算できます.

<details markdown=1><summary>Example</summary>
``` python
--8<-- "code/mcmc/try_mhmcmc_normal.py"
```
![生成されたデータの散布図](img/try_mhmcmc_normal.png)
</details>


__3. 上記の正規分布に $x_{[1]} < (x_{[0]}+1)^2+1$ という不等式制約を加えてください.__

不等式制約を満たさなければ大きな負の値を加えるように対数尤度関数を修正してください. 確率分布は規格化されていなくても問題ありません.

<details markdown=1><summary>Example</summary>
``` python
--8<-- "code/mcmc/try_mhmcmc_conditional.py"
```
![生成されたデータの散布図](img/try_mhmcmc_conditional.png)
</details>


## 各クラスの定義

参考までに {==MHMCMCSampler==} と {==GaussianStep==} の定義全体を以下に示しました. 必要に応じて機能を付け加えるなどの改造をしてみてください.

### MHMCMCSampler

``` python
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
```


### GaussianStep

``` python
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
```

[mhmcmc]: https://raw.githubusercontent.com/FoPM-Astronomy-UTokyo/course/main/code/mcmc/mhmcmc.py
