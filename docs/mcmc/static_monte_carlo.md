# 静的なモンテカルロ法

ここではマルコフ連鎖ではないモンテカルロ法を用いて計算をする方法とその限界について触れます. マルコフ連鎖によって動的にサンプルを生成する方法と対比するために, ここでは静的なモンテカルロ法と呼ぶことにします.[^1]

[^1]: この呼び方は『[統計科学のフロンティア 12 計算統計II マルコフ連鎖モンテカルロ法とその周辺][iba]』に倣いました.

[iba]: https://www.amazon.co.jp/dp/400730789X


## 静的なモンテカルロ法による期待値計算

[はじめに](./index.md) で説明したように, ある確率分布関数 $P(x)$ に従う確率変数 $x$ から計算される量 $A(x)$ の期待値を知りたければ, 確率分布関数 $P(x)$ に従う乱数列 $\{x_i\}_{i=1{\ldots}n}$ を生成して平均値を計算するという手法が使えます.

ここで, 以下のような問題設定を考えます.

1. 逆関数法のように変数変換によって $P(x)$ に従う乱数を生成することはできない.
1. $P(x)$ は規格化定数を除いて容易に計算できる. 別の言い方をすると, $P(x)$ と $P(x')$ の比は容易に計算できるが, $P(x)$ の絶対的な値はわからない.

前者については, 解析的な解があればそれを使えばいいという話なので特に解説の必要はないと思います. また, Bayes の定理に基づいて立式すると後者のようなケースに度々遭遇します. 実験・観測によってデータ $D$ を得たときにパラメタが $\alpha$ である確率は

$$
P(\alpha | D) = \frac{1}{Z}P(D | \alpha)P(\alpha),
$$

と書けます. ここで $Z$ は規格化定数です. $Z$ を求めるためには右辺を $\alpha$ について積分して 1 になるように定めれば良いわけですが, $\alpha$ が多次元の場合には積分の計算コストが極めて高くなります. $Z$ を直接評価することなしに $P(\alpha | D)$ に従う確率変数 $\alpha$ をサンプリングできれば, 計算コストを大きく抑えることができます.

[乱数を生成する](./generate_random_variables.md) で紹介した棄却法はまさにそのような手法のひとつでした. 以下では棄却法と似た考え方に基づいて, 提案分布 $Q(x)$ を経由して $A(x)$ の期待値を求めるための手続きを例示します. ただし $\tilde{P}(x)$ は $P(x)$ から規格化定数 $Z$ を除いた関数とします.[^2]

[^2]: つまり $P(x) = \tilde{P}(x)/Z$ です.

1. 提案分布 $Q(x)$ から確率変数 $\{x_i\}_{i=1{\ldots}n}$ をサンプルする.
2. 確率変数 $x_i$ に対する重み $w_i = \tilde{P}(x_i)/Q(x_i)$ を計算する.
3. 以下の式によって $A(x)$ の期待値と規格化定数を得る.

$$
\left\langle{A(x)}\right\rangle_{x{\sim}P(x)}
= \frac{\sum_i w_i A(x_i)}{\sum_i w_i},
\qquad
Z = \frac{1}{n}\sum_i w_i.
$$

棄却法では $w_i$ に比例する確率で $x_i$ を採用していましたが, ここでは棄却せずに $w_i$ をサンプルされたデータに対する重みとして扱います.[^3][^4]

[^3]: 期待値を求めるという観点からは, ある確率で採用することはその確率に比例した重みを付けることと同義です. データを予め間引く (棄却法) か期待値計算で寄与を圧縮するか, というタイミングの違いです.

[^4]: 棄却する場合には欲しいサンプル数を得るまで loop を回す必要がありますが, Python など一部のプログラミング言語は loop を回すとパフォーマンスが大きく低下することがあるため, そういった意味でもこちらのほうが便利だったりします.


??? tip "上記の式が正しいことの確認"
    $w_i$ も $x_i$ も提案分布 $Q(x)$ からサンプルされたデータなので, データ数が十分に多いときは, $w_i A(x_i)$ の平均値を $x \sim Q(x)$ に対する期待値で置き換えることができます.

    $$
    \frac{1}{n}\sum_i w_i A(x_i) \simeq
    \left\langle w_i A(x_i) \right\rangle_{x \sim Q(x)}
    = \int \mathrm{d}x \,Z\frac{P(x)}{Q(x)}\,A(x)\,Q(x)
    = Z \left\langle A(x_i) \right\rangle_{x \sim P(x)}.
    $$

    同様の操作を $w_i$ の平均値にも実行すると以下のようになります.

    $$
    \frac{1}{n}\sum_i w_i \simeq
    \left\langle w_i \right\rangle_{x \sim Q(x)}
    = \int \mathrm{d}x \,Z\frac{P(x)}{Q(x)}\,Q(x)
    = Z.
    $$

    よって $x \sim P(x)$ に対する $A(x)$ の期待値が適切に計算されていたことになります.


### 円周率の計算

モンテカルロ法をもちいた計算の例としてよく出されるものに円周率の計算があります. 上記の設定にあわせて以下のように問題を定義してみます.

1. 1 辺の長さが 2 である正方形内部の一様分布を提案分布 $Q(x)$ とする.
1. 半径 1 である円内部の一様分布を $P(x)$ とする.
1. $\tilde{P}(x) = 1$ (円内部) / $0$ (それ以外) とする.
1. このとき $\int\mathrm{d}x\,P(x) = 1$ より規格化定数 $Z$ が正方形に対する円の面積の割合となる.

以下にこの設定に従って計算するコードを示します. $Q(x)$ から 10⁵ 個のデータをサンプリングして半径 1 の円の面積を推定しました. グラフは総データ数と推定された面積の推移を表しています.

``` python
import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
gen = default_rng(2021)

func = lambda x,y: (x**2+y**2 < 1.0)

A = 2*2
N = 100000
n = np.arange(N)+1.0
x = gen.uniform(-1,1,size=(N))
y = gen.uniform(-1,1,size=(N))
w = A*func(x,y)

print(f'estimated area: {w.sum()/N}')

fig = plt.figure()
ax = fig.add_subplot()
ax.semilogx(n, w.cumsum()/n)
ax.semilogx(n, np.pi*np.ones_like(n))
ax.set_ylabel('estimated area')
ax.set_xlabel('number of samples')
plt.tight_layout()
plt.show()
```

??? summary "計算結果"
    estimated area: 3.14508
    ![円周率](img/static_circle_dim2.png)


### N 次元球の体積の計算

上記では 2 次元空間に対して計算をしましたが, まったく同じことを N 次元空間で計算してみましょう. N 次元立方体に内接する N 次元球の体積を考えます. なお N 次元球の体積は以下の式で計算することができます.

$$
V_N = \frac{\pi^{N/2}}{\Gamma(N/2 + 1)}.
$$

$\Gamma(x)$ はガンマ関数です. この解析解に対してどのように収束していくかを調べます. 以下に $N=15$ の計算例と結果を示しました. 先程よりサンプル数を増やして 10⁶ 個のデータを $Q(x)$ からサンプリングしています.

``` python
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.special import gamma
import numpy as np
gen = default_rng(2021)

func = lambda x: ((x*x).sum(axis=1) < 1.0)

N = 15
A = 2**N
M = 1000000
n = np.arange(N)+1.0
x = gen.uniform(-1,1,size=(M,N))
w = A*func(x)
V = np.pi**(N/2)/gamma(N/2+1)

print(f'estimated area: {w.sum()/N:.5f} ({V:.5f})')

fig = plt.figure()
ax = fig.add_subplot()
ax.semilogx(n, w.cumsum()/n)
ax.semilogx(n, V*np.ones_like(n))
ax.set_ylabel('estimated volume')
ax.set_xlabel('number of samples')
plt.tight_layout()
plt.show()
```

??? summary "計算結果"
    ```
    estimated area: 0.22938 (0.38144)
    ```
    ![N 次元球の体積](img/static_circle_dimN.png)


結果は解析的に計算される値に比べてかなり低い値 (60% ほど) になりました. 体積の推移を見てみると, 先程の例と比較して値がまったく収束していないことがわかります. 10⁵ 個を超えるまでは 0 を示しており, $Q(x)$ からサンプルしたデータが 1 つも N 次元球に入らなかったことになります.[^5]

[^5]: N 次元立方体の体積との比をとってみると $N = 15$ のときにおよそ 10⁻⁵ なのでおおよそ確率通りと言えます.

次元数を変えて計算をしてみると $N$ が小さいうちは問題なく収束していきますが, $N = 13$ あたりから急に収束しなくなります. これは N 次元立方体の体積は次元が大きくなるとほとんどを壁際が担うようになるためです.

N 次元空間では提案分布 $Q(x)$ とターゲットとなる分布 $P(x)$ の違いがモンテカルロ法による計算の効率に大きく関わってきます. データの次元 N が大きい場合には, 静的なモンテカルロ法による期待値計算は収束性の面で大きな問題を抱えることになります.
