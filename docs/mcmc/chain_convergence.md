# マルコフ連鎖の収束

マルコフ連鎖モンテカルロ法 (MCMC) では, 生成された状態の列 $\{x_i\}_{i=0{\ldots}}$ が定常分布に収束するまで状態遷移を繰り返して連鎖を伸ばす必要があります. ここでは分布が収束しているか, また状態遷移が効率よく行われているかを調べるための方法をいくつか紹介します.


## トレースを確認する
サンプリングされたデータの値を時系列に表示したものをトレースといいます. トレースを確認することで MCMC が機能している (効率よく状態遷移している) かを大まかに確認することができます.


### 分布が適切に収束した場合

ここでは例として Rosenbrock の関数[^1]を使って以下の確率分布に従う乱数を生成します.[^2]

$$
P(x) \propto \mathrm{e}^{-\left(
  (x_{[0]} - a)^2 + b(x_{[1]} - {x_{[0]}}^2)^2
\right)}
$$

$(a,\,b)$ はそれぞれ $(2.0,\,0.2)$ としました. 以下にサンプリングした結果を散布図として表示するコードのサンプルと結果を示します. ここでは最初の 1000 個を burn-in として捨てて 10⁵ 個のデータをサンプリングしています.

[^1]: バナナ型のポテンシャルを持つ関数で, 数値最適化アルゴリズムの性能評価やチュートリアルでよく使われます. 個の商のサンプルとしてこの関数を採用した深い理由は特にありません.

[^2]: $x$ の $i$ 番目の要素を $x_{[i]}$ で表します.


``` python
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))


step = GaussianStep(5.0)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

x = np.linspace(-6,10,1000)
y = np.linspace(-20,100,1000)
xx,yy = np.meshgrid(x,y)
xy = np.dstack((xx.flatten(),yy.flatten())).T
z = np.log(rosenbrock(xy).reshape(1000,1000))
lv = np.linspace(-2,4,13)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.contour(xx,yy,z,levels=lv, alpha=0.3)
ax.scatter(sample[::20,0], sample[::20,1], marker='.', s=1)
ax.set_xlabel('random variable: x0')
ax.set_ylabel('random variable: x1')
fig.tight_layout()
plt.show()
```

??? summary "計算結果"
    ![サンプリング結果の散布図](img/try_mhmcmc_rosenbrock.png)


Rosenbrock の関数は $(a,\,a^2)$ に最小値を持ちます. 今回は $a=2$ なので散布図で $(2,\,4)$ 付近にサンプリングされたデータが集まっていることがわかります.

`mhmcmc` で提供されている関数 `display_trace()` を使ってトレースを表示します. 以下のコードで黄色でハイライトされている部分を加えてください. 1 列目が $x_{[0]}$ のトレースとヒストグラムを表しています.  2 列目は $x_{[0]}$ のトレースとヒストグラムを表しています.

``` python hl_lines="24 25"
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))


step = GaussianStep(5.0)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

from mhmcmc import display_trace
display_trace(sample)
```

??? summary "計算結果"
    ![サンプリング結果のトレース](img/try_mhmcmc_rosenbrock_trace.png)


トレースは $x_{[0]}$, $x_{[1]}$ どちらも目立った構造はなく帯状に広がっていることが分かります.[^3] 分布が収束していればデータ点の分布は時間に依らないと期待されるため, 大局的にはこのように帯のような見た目になっていると安心できます. また, ヒストグラムのピークは $x_{[0]} = 2$, $x_{[1]} = 4$ 付近にあることも確認できます. 目的とする分布に収束しているだろうと判断できます.

[^3]: この時点ではデータを間引いていないので, より細かく表示させるとランダムウォーク的に動いている様子が見えてきます. 大きなスケールで見ているので潰されているだけです.


### ステップ幅が狭すぎる場合

同じ計算を `GaussianStep` の幅を極端に狭くして動かしてみます. 黄色でハイライトした部分が更新した箇所です. 一度に移動できる距離が短くなったために, 分布が収束するまでに必要な時間が伸びるだろうと期待されます.

``` python hl_lines="15"
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))


step = GaussianStep(0.01)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

from mhmcmc import display_trace
display_trace(sample)
```

??? summary "計算結果"
    ![サンプリング結果のトレース](img/try_mhmcmc_rosenbrock_trace_small.png)


トレースを確認すると $x_{[0]}$, $x_{[1]}$ どちらも帯状にはならず, ランダムウォーク的な挙動が見えています. ヒストグラムも先ほど確認したような形からは大きく外れています. 提案分布の幅が狭すぎるために分布が収束していない, あるいはステップ幅に対して計算時間が足りていないと判断できます.


### ステップ幅が広過ぎる場合

今度は `GaussianStep` の幅を極端に広くして動かしてみます. 黄色でハイライトした部分が更新した箇所です. 一度に長い距離を移動できるようになりました. 一方で, 低確率の状態への遷移を頻繁に提案されるため, ほとんどの状態遷移が却下され, 結果として確率分布の収束は遅くなります.


``` python hl_lines="15"
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))


step = GaussianStep(100.)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]

from mhmcmc import display_trace
display_trace(sample)
```

??? summary "計算結果"
    ![サンプリング結果のトレース](img/try_mhmcmc_rosenbrock_trace_large.png)


トレースを確認すると矩形のラインが確認できます. 状態遷移が却下され続けるため, 位置をほとんど移動することができていないことが分かります. このような場合でも非常に長い時間をかければ定常分布に収束します. しかしながら, このようなトレースが得られた場合には, 一般には計算のセッティングを見直すことが必要だと思われます.


## 平均値の検定

### 平均値の検定: t-検定

データ列の前半と後半の平均値の差を分布が収束したかどうかを判定する基準として使うことができます.t-検定は 2 つのデータセットの平均値が等しいかどうかを判定する手法です. 先ほどのデータに対して t-検定を適用してみます. 黄色でハイライトした部分が主な変更箇所です. 前半部分と後半部分からそれぞれ 20% ずつ採用して比較します.

`stats.ttext_ind(a,b)` は t-検定に基づいて 2 つのデータセットの平均値が等しいとみなせる確率を返す関数です. 下のサンプルでは `pv` という変数に確率が格納されています.

``` python hl_lines="15 23-31"
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))


step = GaussianStep(5.0)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000:]
m,n = sample.shape
m20,m50 = int(m/5), int(m/2)

from scipy import stats
tv,pv = stats.ttest_ind(
  sample[:m20,:], sample[m50:(m50+m20),:], equal_var=False)

print(f'probability(x0[former] == x0[latter]): {pv[0]:g}')
print(f'probability(x1[former] == x1[latter]): {pv[1]:g}')
```

??? summary "計算結果"
    ```
    probability(x0[former] == x0[latter]): 0.0141046
    probability(x1[former] == x1[latter]): 2.97703e-05
    ```

上記のサンプルでは計算結果はほぼゼロ (平均値が同じだとはみなせない) でした. t-検定の結果は収束したとは言えないというものになりました.[^4]

[^4]: サンプリング数が足りず本当に収束していないのでは？と思うかもしれません. 試しに数を 5 倍に増やしてみたところ確率はさらに減りました. サンプル数が増えることによって差がより有意になったと判定されたようです.

t-検定ではそれぞれのデータセットに含まれるデータは独立に選ばれたものであるという前提があります. しかしながら, マルコフ連鎖によって生成されたデータセットは直前の状態と強く相関しています. そのため, 独立なサンプリングだとみなせるデータ数は実際にサンプリングしたデータ総数よりも必然的に減っています.

上記の例では全データを 2 つに分けてそのまま平均値の差を検定しました. そのため, 独立なデータ数を不当に水増しして t-検定を実施したことになっています.


### 自己相関関数

データセットがどの程度の相関を持っているかを評価する方法に自己相関関数があります. 自己相関関数はデータの時刻を $k$ だけシフトさせて, 元のデータとの相関をとることで計算します.

$$
{\operatorname{autocorr}}\left(k; \{x_i\}_{i=0{\ldots}n}\right)
= \sum_i (x_i - \bar{x}) \, (x_{i+k} - \bar{x})
\quad (-n \leq k \leq n).
$$

以下に自己相関を計算するための Python 関数のサンプルを示します.

``` python
def autocorrelation(x: np.ndarray) -> np.ndarray:
  ''' Calculate auto-correlation of x

  Parameters:
    x (numpy.ndarray): Data in (M,N)-array, where M is the number of samples
        and N is the number of dimensions.

  Returns:
    numpy.ndarray: Calculated auto-correlation in a (M,N)-array
  '''
  if x.ndim==1: x = np.expand_dims(x,axis=0)
  v = x - x.mean(axis=0)
  M, N = v.shape
  var  = v.var(axis=0)
  corr = []
  for n in range(N):
    corr.append(np.correlate(v[:,n],v[:,n],mode='full')/var[n])
  return np.arange(-(M-1),M), np.stack(corr).T/M
```

ただし, このサンプルでは $k=0$ のときの値 ($\{x_i\}_{i=0{\ldots}}$ の分散) で規格化しています. また, この関数も `mhmcmc.py` に含まれているので `import` して使うことができます.

正規分布に従う乱数であれば相関がないことが期待されます. ここでは `numpy` によって提供されている正規分布関数に従う乱数を使って `autocorrelation` の挙動を確かめてみます.

``` python
from mhmcmc import autocorrelation
from numpy.random import default_rng


gen = default_rng(2021)
data = gen.normal(0, 1, size=(100))
k, corr = autocorrelation(data)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot()
ax.plot(k, corr, marker='.')
ax.set_xlabel('displacement: k')
ax.set_ylabel('autocorr for N(0,1)')
fig.tight_layout()
plt.show()
```

??? summary "計算結果"
    ![標準正規分布の自己相関関数](img/sample_autocorr.png)


相関がなければ自己相関関数の値は 0 になります. 標準正規分布に従う乱数は相関を持たないため $k=0$ でのみ 0 でない期待値を持ちます.

それでは MCMC によって生成されたデータ列にこの関数を適用してみましょう.

``` python hl_lines="2 22-37"
from mhmcmc import MHMCMCSampler, GaussianStep
from mhmcmc import autocorrelation
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))

step = GaussianStep(5.0)
x0 = np.zeros(2)
model = MHMCMCSampler(log_likelihood, step)
model.initialize(x0)
sample = model.generate(101000)
sample = sample[1000:,:]

k, corr = autocorrelation(sample)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(k, corr[:,0], marker='')
ax1.set_xlim([-800,800])
ax1.set_ylabel('autocorr for x0')
ax2 = fig.add_subplot(2,1,2)
ax2.plot(k, corr[0,1], marker='')
ax2.set_xlim([-800,800])
ax2.set_ylabel('autocorr for x1')
ax2.set_xlabel('displacement: k')
fig.tight_layout()
plt.show()
```

??? summary "計算結果"
    ![自己相関関数](img/try_mhmcmc_rosenbrock_autocorr.png)


出力されたグラフを見ると $k=0$ 以外にも有意に 0 でない値を持つことが分かります. このケースでは $x_{[0]}$ でも, $x_{[1]}$ でも, おおよそ $k=200$ 程度でほぼ 0 になります. つまり, データをすべて使うのではなく 200 個程度おきに使うことで, 相関がほぼ 0 の乱数とみなせると期待できます.


### 平均値の検定: t-検定 (もう一度)

以上のことをふまえて, もう一度 t-検定を実施してみます. サンプリング総数は 10⁵ 個ですが, 200 個おきにデータを採用するので独立なサンプルとみなせる総データ数は 500 個です. これを 2 つに分けて平均値が等しいかどうかを t-検定で検証します.


``` python hl_lines="22"
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))


step = GaussianStep(5.0)
model = MHMCMCSampler(log_likelihood, step)

x0 = np.zeros(2)
model.initialize(x0)

sample = model.generate(101000)
sample = sample[1000::200]
m,n = sample.shape
m20,m50 = int(m/5), int(m/2)

from scipy import stats
tv,pv = stats.ttest_ind(
  sample[:m20,:], sample[m50:(m50+m20),:], equal_var=False)

print(f'probability(x0[former] == x0[latter]): {pv[0]:g}')
print(f'probability(x1[former] == x1[latter]): {pv[1]:g}')
```

??? summary "計算結果"
    ```
    probability(x0[former] == x0[latter]): 0.190402
    probability(x1[former] == x1[latter]): 0.966328
    ```

前回と大きく変わって, 平均値が同じ分布からのサンプリングだとみなせる確率がそれぞれ 19%, 96% という結果になりました. 前半と後半で有意に違うとはみなせない程度の値になっています. 今回の例のように, サンプリングしたデータ間の相関が影響を与える場合には, データを間引いて独立したサンプリングだとみなせるデータだけを使用する必要があります.[^5]

[^5]: ちなみに期待値 (単純平均) の計算ではデータの相関は影響しません (収束がデータ数のわりには遅くなるだけ).


## 連鎖内-連鎖間分散の計算

時間が許せば MCMC を複数回実行して比較をするという方法が使えます. それぞれのマルコフ連鎖内部の分散 (within-chain) に比べて, 複数のマルコフ連鎖間の分散 (between-chain) が大きい場合には分布が収束していないとみなすことができます.

合計 $M$ 回の MCMC を実行し, それぞれの chain で $N$ 個のデータをサンプリングしたとします. このとき,  within-chain variance $\sigma^2_w$ と between-chain variance $\sigma^2_b$ は以下のように書けます.

$$
\left\{~\begin{aligned}
\sigma^2_w &=
\frac{1}{M(n-1)}\sum_{m=1}^M\sum_{i=1}^n (x_{m,i} - \bar{x}_{m})^2 \\
\sigma^2_b &= \frac{n}{M-1}\sum_{m=1}^M (\bar{x}_m - \bar{x})^2.
\end{aligned}\right.
$$

ここで $\bar{x}_m$ は各 chain 内での平均値, $\bar{x}$ はデータ全てに対する平均値です. ここで全体の分散[^6][^7]と between-chain variance $\sigma^2_b$ の比の平方根として $\hat{R}$ を定義します.

[^6]: 全体の分散の "推定値" として $\sigma^2_{bw} = \left((n-1)\sigma^2_w + \sigma^2_b\right)/n$ という式を使っているようです. Gelman et al. (2013) によると "This quantity _overestimates_ the marginal posterior variance assuming the starting distributions in appropriately overdispersed, but is _unbiased_ ...". とのことでした.

[^7]: 手元で全体の分散を計算をしてみたら $\sigma^2_{bw} = \left((n-1)\sigma^2_w + (1-M^{-1})\sigma^2_b\right)/(n-M^{-1})$ になりました. ただし, データの相関については何も考慮していません. $\sigma^2_w$ と $\sigma^2_b$ の重み付き平均になるという結果は確認できました.

$$
\hat{R} = \sqrt{1 + \frac{\sigma^2_b - \sigma^2_w}{n\sigma^2_w}}.
$$

この量は $\sigma^2_b = \sigma^2_w$ のときに $\hat{R} \simeq 1$ となります. 慣習として $\hat{R} \lesssim 1.1$--$1.2$ ほどであれば収束したと考えてよいということになっています.

$\hat{R}$ をつかって収束したと言えるかどうかを判定してみます. 以下に 4 つの MCMC chains を生成して $\hat{R}$ を計算するサンプルを示します.


``` python
from mhmcmc import MHMCMCSampler, GaussianStep
import numpy as np


a = 2.0
b = 0.2

def rosenbrock(x):
  return (x[0]-a)**2 + b*(x[1]-x[0]**2)**2

def log_likelihood(x):
  return -(rosenbrock(x))

nchain = 4
step = GaussianStep(5.0)
samples = []
for n in range(nchain):
  x0 = np.zeros(2)+n
  model = MHMCMCSampler(log_likelihood, step)
  model.initialize(x0)
  sample = model.generate(101000)
  samples.append(sample[1000::])

samples = np.stack(samples)

nsample = samples.shape[1]
intra_mean  = samples.mean(axis=1)
global_mean = samples.mean(axis=(0,1)).reshape((1,2))
within_var  = samples.var(axis=1).mean(axis=0)
between_var = nsample*np.var(intra_mean-global_mean,axis=0)
Rhat = np.sqrt(1 + (between_var/within_var-1)/nsample)
print(f'Rhat values: {Rhat}')
```

??? summary "計算結果"
    ```
    Rhat values: [1.00044594 1.00048218]
    ```

計算した $\hat{R}$ の値はほぼ 1 に近く, 十分に収束したと判定してもよさそうです.
[^8]

[^8]: ただしパラメタを変えて計算してみると $\hat{R} \simeq 1$ の場合でもトレースを見るとまったく収束していないというケースもありました. 重要な計算では過信せずにトレースや相関も確認しておいたほうがよいと思います.
