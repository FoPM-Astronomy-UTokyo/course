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

[^1]: バナナ型のポテンシャルを持つ関数で, 数値最適化アルゴリズムの性能評価やチュートリアルでよく使われます. この章で特にこの関数を使う意味はありません.

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


## 自己相関関数
## 連鎖内-連鎖間分散
