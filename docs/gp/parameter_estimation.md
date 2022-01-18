# カーネルパラメタの推定

この章ではガウス過程を活用する最初のステップとして, カーネルのパラメタを推定する方法を紹介します. カーネルをデータセットにあわせて最適化することで, データセットが内在する構造 (なめらかさ) をカーネルに覚えさせます.


## データセットの作成
まずは推定の対象となるデータセットを用意します. データの背後にある関係式 (ground truth) を以下の式で定義します.

$$
  y = 0.5\sin(3x) + \varepsilon, \quad
  \varepsilon \sim \mathcal{N}(0.0, 0.2).
$$

測定点 $\{X_i\}$ は $[0, 5)$ から一様に 50 個サンプルします. 上記の式に従って $\{Y_i\}$ を生成します. $\{X_i\}$ をサンプルするときには `dist.Uniform()` を, $\{Y_i\}$ をサンプルするときには (測定ノイズとして) `dist.Normal()` を使用しました.

``` python
import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)

import numpyro
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist

key_x,key_y,key_mc = random.split(random.PRNGKey(42),3)

M = 50
X = jnp.sort(dist.Uniform(0,5).expand([M,]).sample(key_x))
Y = 0.5*jnp.sin(3*X) + dist.Normal(0.0,0.2).expand([M,]).sample(key_y)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot()
ax.plot(X, Y, 'kx')
ax.set_xlabel('$X$', fontsize=16)
ax.set_ylabel('$Y = 0.5\sin(3X)+\epsilon$', fontsize=16)
fig.tight_layout()
plt.show()
```
![データセット](./img/inference_dataset.png)


## カーネルの選択
データセットは周期的な関数から生成しましたが, ここでは背後にある関係式は知らなかった, ということにしましょう. プロットした散布図を見ると, 大きいスケールでなめらかに変動しているので [Radial Basis Function Kernel (RBF Kernel)][rbf] と相性が良さそうです. そこで, この章では RBF Kernel と [Noise Kernel][noise] の和でデータセットを表現してみます. RBF Kernel と Noise Kernel の定義はそれぞれ以下のとおりです.[^1]

[^1]: ここでは Noise Kernel を単位行列 (`jnp.eye()`) を使って定義しています. [前の章][noise]で定義したようにデルタ関数的に定義しても問題ありませんが, 数値誤差に注意する必要があります.

[rbf]: ./different_kernels.md#radial-basis-function-kernel
[noise]: ./different_kernels.md#noise-kernel

``` python
def rbf_kernel(X, Z, var, length):
  dXsq = jnp.power((X[:, None] - Z) / length, 2.0)
  k = var * jnp.exp(-0.5*dXsq)
  return k

def noise_kernel(X, Z, noise):
  nx,nz = X.size,Z.size
  return noise*jnp.eye(nx,nz)
```

上記のカーネル関数を使用してデータセットを再現するモデルを作成します. ここでは確率的プログラミング言語の `numpyro` の作法に従って関数 `model()` を作成します. まずは以下の定義式を見てください.

``` python
def model(X, Y):
  var    = numpyro.sample('variance', dist.LogNormal(0.0, 3.0))
  length = numpyro.sample('lengthscale', dist.LogNormal(0.0, 3.0))
  noise  = numpyro.sample('noise', dist.LogNormal(0.0, 3.0))

  K = rbf_kernel(X, X, var, length) + noise_kernel(X, X, noise)

  numpyro.sample(
      'y',
      dist.MultivariateNormal(
          loc=jnp.zeros(X.shape[0]),
          covariance_matrix=K
      ),
      obs=Y,
  )
```

モデルは引数としてデータセット $\{X_i,Y_i\}_{i=1{\ldots}M}$ をとります. モデルは最適化すべきパラメタとして `variance`, `lengthscale`, `noise` を持っています. 最初の 3 行でそれぞれのパラメタを宣言しています. 詳細については後述します.

5 行目以降では, パラメタと説明変数 $\{X_i\}$ を使用してカーネルを定義し, 確率変数 `y` が平均 0 で共分散行列が `K` である多次元正規分布に従うことを宣言しています. 数式で表現すると以下のとおりです. Likelihood の定義に対応しています.

$$
p(y \,|\, x,\sigma^2,l,\varepsilon^2) = \mathcal{MultiN}(0, \Sigma), \quad
\Sigma = K(x,x; \sigma^2, l, \varepsilon^2).
$$

$\sigma^2$, $l$, $\varepsilon^2$ はそれぞれ `variance`, `lengthscale`, `noise` に対応しています. また, 最後の `obs=Y` というキーワードで, 観測したデータセット $\{Y_i\}$ の確率分布が `y` に従うことを宣言しています.

データセット $\{X_i, Y_i\}$ が観測されたときの $\sigma^2$, $l$, $\varepsilon^2$ の事後確率を最大化することでパラメタを最適化できます. 先ほど定義した Likelihood に加えて $\sigma^2$, $l$, $\varepsilon^2$ の事前確率があれば Bayes の定理にしたがって事後確率を得られます.

$$
\begin{aligned}
p(\sigma^2,l,\varepsilon^2 \,|\, x, y)
&\propto p(y \,|\, x, \sigma^2, l, \varepsilon^2)
         \,p(\sigma^2, l, \varepsilon^2) \\
&\sim p(y \,|\, x, \sigma^2, l, \varepsilon^2)
      \,p(\sigma^2)\,p(l)\,p(\varepsilon^2)
\end{aligned}
$$

このモデルでは各パラメタは独立に決まっていると仮定しましょう (2 行目).

現時点で各パラメタを推定するための事前情報はありません. ただしいずれのパラメタも正の値をとるため, 事前分布として[幅の広い対数正規分布][lognormal]を仮定しておくことにします. 正の値であるという以外はほとんど制約がない無情報事前分布になります.

[lognormal]: https://www.wolframalpha.com/input/?i=log+normal+distribution+%280.0%2C+3.0%29

以上が `model()` 関数の説明です. 独特で少々ややこしい書式ですが, この作法でモデルを定義しておくことで `numpyro` の強力な最適化機能を使うことができます.


## パラメタの最適化
いよいよパラメタの最適化をおこないます. ここではモデル関数で定義した事後確率分布からマルコフ連鎖モンテカルロ法 (Markov Chain Monte Carlo; MCMC)[^2] をもちいてパラメタ $(\sigma^2, l, \varepsilon^2)$ をサンプリングします. サンプルコードを以下に示します.

[mcmc]: ../mcmc/index.md
[^2]: マルコフ連鎖モンテカルロ法についての解説は 2020 年の[実習テキスト][mcmc]を参考にしてください. なお, ここではハミルトニアンモンテカルロ法 (Hamiltonian Monte Carlo; HMC) と No-U-Turn Sampler (NUTS) という手法を使ってサンプリングを高速化しています.

``` python
from numpyro.infer import NUTS, MCMC

sampler = NUTS(model)
mcmc = MCMC(sampler,
    num_warmup=500, num_samples=1000, num_chains=1, progress_bar=True)
mcmc.run(key_mc, X, Y)
mcmc.print_summary()
```

`numpyro.infer` から `NUTS` と `MCMC` というクラスをインポートして使用します.  `mcmc` というオブジェクトが計算を実行する本体です. 最初に 500 回ウォームアップ計算をしたあと, `model()` で定義した事後確率分布から 1000 個のパラメタをサンプリングします.

計算が完了すると計算結果の要約が表示されます. もし, 下に示したサンプルに比べて `r_hat` が大きかったり `n_eff` が少なかったりする場合は, モデル関数ががうまく定義できていない可能性があります. 関数の定義を見直してみてください.

```
                 mean       std    median      5.0%     95.0%     n_eff     r_hat
lengthscale      0.54      0.16      0.53      0.29      0.80    384.93      1.00
      noise      0.04      0.01      0.04      0.03      0.05    495.49      1.00
   variance      0.39      0.63      0.21      0.04      0.81    306.38      1.01
```

??? tips "計算過程のもう少し詳しい説明"
    マルコフ連鎖モンテカルロ計算では, 現在の状態 (パラメタ) から次の状態 (パラメタ) へと確率的に移り変わることで事後確率分布に従うパラメタをサンプリングします. このとき, 現在地点から次に移動する状態を提案する関数をサンプラと呼びます.

    上記のコードではハミルトニアンモンテカルロ法 (Hamiltonial Monte Carlo; HMC) を使用しています. この手法では事後確率分布をポテンシャルとみなして, その中で質点が運動する過程をシミュレートすることで, 効率よくパラメタ空間を探索することができます. このとき, 現在地点から出発した質点がめぐりめぐってもとの位置に戻ってきてしまうと計算がまるまる無駄になってしまいます. そこで No-U-Turn Sampler (NUTS) というサンプラを使用します. NUTS は HMC のシミュレーション計算で無駄が起こりにくいようにやめ時を与えてくれるアルゴリズムです.

    HMC はとても強力な手法ですが, 事後確率分布関数をパラメタで微分した関数が必要になります. 本来はかなり準備が必要な手法なのですが, `jax` や `numpyro` というツールがややこしいところを覆い隠してくれています.

MCMC でサンプリングした結果は `get_samples()` という関数で取得できます.

``` python
sample = mcmc.get_samples()

sqrtN = jnp.sqrt(sample['variance'].shape[0])
print('variance    = {:.4f}+/-{:.4f}'.format(
    sample['variance'].mean(),sample['variance'].std()/sqrtN))
print('lengthscale = {:.4f}+/-{:.4f}'.format(
    sample['lengthscale'].mean(),sample['lengthscale'].std()/sqrtN))
print('noise       = {:.4f}+/-{:.4f}'.format(
    sample['noise'].mean(),sample['noise'].std()/sqrtN))
```
??? summary "計算結果"
    ``` python
    variance    = 0.3868+/-0.0199
    lengthscale = 0.5416+/-0.0050
    noise       = 0.0398+/-0.0003
    ```


## カーネルの推定
以上でカーネルの推定に必要な情報は出揃いました. ここでは MCMC によってサンプリングされた 1000 個のパラメタの平均値を採用することにします.

``` python
var    = sample['variance'].mean()
length = sample['lengthscale'].mean()
noise  = sample['nose'].mean()
```

最適化したカーネルをもちいてデータを生成してみましょう. `model()` 関数と同じカーネルを定義して `MultivariateNormal()` を使用してデータをサンプリングします. ここでは 3 つのデータ系列をサンプリングしました.

``` python
key = random.PRNGKey(42)

K = rbf_kernel(X, X, var, length) + noise_kernel(X, X, noise)
z = dist.MultivariateNormal(covariance_matrix=K).expand([3,]).sample(key)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot()
ax.plot(X, Y, 'kx')
ax.plot(X, z.T, '.-', lw=1)
ax.set_xlabel('$X$', fontsize=16)
ax.set_ylabel('$Y = 0.5\sin(3X)+\epsilon$', fontsize=16)
fig.tight_layout()
plt.show()
```

サンプルしたデータ (青/緑/黄) を最初のデータセット (黒) と一緒にプロットしました. サンプルしたデータと最初のデータセットが似ている, つまりノイズの大きさや変動の典型的なスケールが同程度であればカーネルはうまく最適化できています.[^3]

[^3]: カーネルが学習したのはデータセットの __なめらかさだけ__ だけです. サンプルしたデータ系列が最初データセットをフィットしているわけではないことに注意してください.

![最適化されたカーネルからサンプルしたデータ系列](./img/inference_optkernel.png)


## カーネルパラメタによる分類

与えられたデータに対してカーネルのパラメタを推定する問題に取り組んでみましょう.


??? Summary "Example"
    ``` python
    --8<-- "code/gp/excercise_kernel_param.py"
    ```
    ![MCMC によるカーネルパラメタの散布図](./img/excercise_kernel_param.png)
