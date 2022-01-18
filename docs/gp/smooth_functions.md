# なめらかな関数

[はじめに](./index.md) ではある種の系列データを表現するときになめらかな関数が役に立つということを述べました. ガウス過程ではなめらかな関数を表現するためにカーネル (共分散) を活用します. ここでは, なぜカーネル (共分散) がなめらかな関数を表現するときに役立つのかを簡単に解説します.

## 多項式によるフィッティング
なめらかに変化する関数として最も単純なものは (低次の) 多項式からなる関数でしょう. なめらかな変動をモデル化するときに多項式で十分なケースは数多くあると思います. 一方で, 多項式によるモデリングには以下の欠点があります.

1. 式の次数を予め決めなければいけない.
1. パラメタ (係数) と「なめらかさ」の関係がわかりにくい.

多項式の次数は一般に関数の極値の数と対応しています.  2 次関数であれば 1 つ, 3 次関数であれば (最大) 2 つ,  $n$ 次関数であれば (最大) $n{-}2$ 個の極値を持ちます. 式の次数は「なめらかさ」ではなく系列が何回蛇行するかということと密接に関わっており, 同様の「なめらかさ」を持ったデータ系列であっても, 極値の数が異なれば違う次数で表現しなければなりません. また, 多項式によって表現される関数がどの程度「なめらか」になるかは極値の位置関係に依存します. ある 2 つのデータ系列を多項式でフィットしたときに「なめらかさ」が近いかどうかを多項式の係数から知ることは用意ではありません.

ガウス過程ではカーネル関数を用いて関数の「なめらかさ」を直接コントロールします. そのため, データ系列が (一様に)[^1] なめらかな性質を持っているのであれば, 系列の変動によらずデータ系列を表現することができます. また, 最適化したカーネルのパラメタが異なるデータ系列間で似ていれば, そのデータ系列は「なめらかさ」という点で類似していると言えます. このようにガウス過程は多項式によるフィッティングに対してユニークな長所を持っています.

[^1]: ガウス過程ではデータ系列をひとつの「カーネル関数」をもちいて表現します. データ系列の中で「なめらかさ」が大きく変動するようなケースでは, ガウス過程でモデル化するには何らかの工夫が必要になるでしょう.


## 2 つの確率変数の相関
まずは簡単なケースとして相関のない 2 変数から始めましょう. 平均が $0$ で分散が $1$ である正規分布からデータ $Y_1$, $Y_2$ をサンプルします.

$$
  Y_1 \sim \mathcal{N}(0, 1), ~~
  Y_2 \sim \mathcal{N}(0, 1).
$$

Python の `numpyro` を使用すると以下のように記述できます.[^2]

``` python
from jax.config import config
config.update('jax_enable_x64', True) # 64 bit 浮動小数点で計算する

import jax.random as random
import numpyro.distributions as dist

key1,key2 = random.split(random.PRNGKey(42))

y_1 = dist.Normal(0,1).sample(key1) # 標準正規分布から 1 つサンプル
y_2 = dist.Normal(0,1).sample(key2)
```
??? success "計算結果"
    ``` python
    (DeviceArray(0.64917078, dtype=float64),
     DeviceArray(1.70584208, dtype=float64))
    ```

[^2]: 最後の 2 行は以下のように記述することもできます.
``` python
y_1 = random.normal(key1) # 標準正規分布から 1 つサンプル
y_2 = random.normal(key2)
```

データの数を増やして散布図を作成してみます.

``` python
key1,key2 = random.split(random.PRNGKey(42))

N = 1000
y_1 = dist.Normal(0,1).expand((N,)).sample(key1)
y_2 = dist.Normal(0,1).expand((N,)).sample(key2)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax.scatter(y_1, y_2, marker='x')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_xlabel('Random Variable $Y_1$', fontsize=16)
ax.set_ylabel('Random Variable $Y_2$', fontsize=16)
plt.show()
```
??? success "計算結果"
    ![相関のない 2 変数](./img/random_no_correlation.png)

2 変数 $Y_1$, $Y_2$ に相関がないため, どちらの軸にも偏らない散布図が作成されました. 念のために生成したデータ $Y_1$ と $Y_2$ の間に相関がないことを確認してみましょう. データの共分散は `jax.numpy.cov(x,y)` で計算できます. ほぼ単位行列に近い結果が得られたと思います.

``` python
print('Covariance of (Y_1, Y_2):')
print(jnp.cov(y_1,y_2))
```
??? success "計算結果"
    ``` python
    Covariance of (Y_1, Y_2):
    [[ 0.95353364 -0.00836174]
     [-0.00836174  1.00543691]]
    ```


次に変数 $Y_1$, $Y_2$ の線型結合によって新しいデータを作成してみます.

``` python
import jax.numpy as jnp
key1,key2 = random.split(random.PRNGKey(42))

N = 1000
y_1 = dist.Normal(0,1).expand((N,)).sample(key1)
y_2 = dist.Normal(0,1).expand((N,)).sample(key2)

z_1 = (y_1 + 0.5*y_2)/jnp.sqrt(1.25)
z_2 = (y_1 - 0.5*y_2)/jnp.sqrt(1.25)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax.scatter(z_1, z_2, marker='x')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_xlabel('Random Variable $Z_1$', fontsize=16)
ax.set_ylabel('Random Variable $Z_2$', fontsize=16)
plt.show()
```
??? success "計算結果"
    ![相関のある 2 変数](./img/random_with_correlation.png)

今度は 2 変数に相関があるため, 散布図は $Z_1 = Z_2$ の直線に沿って偏った分布をしています. $Z_1$ が大きいときには $Z_2$ も大きくなるという傾向を反映しています.

先ほどと同様に $Z_1$ と $Z_2$ の共分散行列を計算してみましょう. 今度は単位行列とは大きく違う結果になるはずです.

``` python
print('Covariance of (Z_1, Z_2):')
print(jnp.cov(z_1,z_2))
```
??? success "計算結果"
    ``` python
    Covariance of (Z_1, Z_2):
    [[0.97060369 0.56173953]
     [0.56173953 0.9572249 ]]
    ```

このように共分散行列の非対角成分の大きさは, データの相関の強さを表すことができます.

標準正規分布からサンプルした 2 つの確率変数 $y = (Y_1, Y_2)^T$ にある行列 $T$ を作用させて $z = T^Ty = (Z_1, Z_2)^T$ に変換することを考えます.[^2] ここで $z$ の共分散行列を $\Sigma$ で表します. すると $\Sigma$ は以下の関係式を満たします.

$$
\Sigma = z z^T = (T^Ty)(T^Ty)^T = T^T(yy^T)T = T^TT.
$$

[^2]: 簡単のため $z$ の平均値は $0$ とします.

このように変換行列 $T$ から $z$ の共分散行列を計算できます. 先程の例で使用した変換行列は以下のとおりです.

$$
T^T = \frac{1}{\sqrt{1.25}}
\begin{pmatrix}
  1 & \phantom{-}0.5 \\
  1 & -0.5
\end{pmatrix}
$$

この行列を使用して共分散行列を計算すると以下のようになります. 乱数を使用して検算した結果とほぼ一致したと思います.

$$
T^T T =
\frac{1}{1.25}
\begin{pmatrix}
  1 & \phantom{-}0.5 \\
  1 & -0.5
\end{pmatrix}
\begin{pmatrix}
  1 & 1 \\
  0.5 & -0.5
\end{pmatrix}
=
\begin{pmatrix}
  1 & 0.6 \\
  0.6 & 1
\end{pmatrix}
$$

今度は逆に共分散行列 $\Sigma$ が与えられたときに, 共分散が $\Sigma$ になるような確率変数を生成してみましょう. 共分散行列は定義より実数の正定値対象行列 (positive definite symmetric matrix) になります. そのため, [コレスキー分解][cholesky] (Cholesky Decomposition) を適用することで, 行列の積に分解することができます.

[cholesky]: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.cholesky.html

$$
\Sigma = L^T L
$$

ここでは $L^T$ を変換行列と見立てて変換をしてみましょう.

``` python
from jax.scipy.linalg improt cholesky
S = jnp.array([[1.0, 0.6],[0.6, 1.0]])
L = cholesky(S)
print(L.T)
y = jnp.stack([y_1, y_2])
z = L.T@y
print(jnp.cov(z))
```
??? success "計算結果"
    ``` python
    [[1.  0. ]
     [0.6 0.8]]
    [[0.95353364 0.56543079]
     [0.56543079 0.97872446]]
    ```
    ![共分散行列から生成した相関のある 2 変数](./img/random_with_correlation-2.png)

変換後の確率変数 `z` の共分散行列が設計した共分散行列 `S` と同等であることを確認してください. なお, コレスキー分解からえられた変換行列 $L^T$ は先ほどの変換で使用したものと異なっています. 標準正規分布にしたがう確率変数から共分散が $\Sigma$ になるような確率変数への変換には自由度があることを反映しています.[^3]

[^3]: $L$ と $T$ はユニタリ行列によって変換することができます.


## 多数の確率変数の相関
ここまでは 2 変数だけに限定して議論をしてきました. 解説した相関のある確率変数を生成する方法はより高次元のデータに対しても同様に適用することができます. 以下では 1000 次元の標準正規分布をサンプルしてみます. このデータには相関がないので共分散行列は 1000 次元の単位行列になります.

``` python
M = 1000 # 次元数
key = random.PRNGKey(42)

y = dist.Normal(0,1).expand([M,]).sample(key)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()
ax.plot(y, 'x')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_xlabel('Dimension: $M$', fontsize=16)
ax.set_ylabel('Random Variable $Y$', fontsize=16)
plt.show()
```
??? success "計算結果"
    ![相関のない 1000 次元データ](./img/random_1000dim_no_correlation.png)

2 変数のときと同様に相関のあるデータを生成してみます. ここでは相関行列 $\Sigma$ を以下のように定義してみます.

$$
\Sigma_{m,n} = \exp\left(
  -\left( \frac{m-n}{l} \right)^2
  \right) + \varepsilon I_{m,n}
$$

$m$, $n$ はそれぞれ行番号と列番号に対応しています. $m{=}n$ であれば相関は $1$ になり, $m$ と $n$ の差が大きくなるほど相関が低下します. $l$ はどれだけ離れたデータまで相関が値を持つかをコントロールするパラメタ (相関長) です. 第 2 項はコレスキー分解の数値安定性のために加えている単位行列です. $l=30$, $\varepsilon=10^{-10}$ として計算するコードを以下に示します.

``` python
M = 1000
n = jnp.arange(M)
d = jnp.power((n[:,None]-n)/30.0, 2)
S = jnp.exp(-d) + 1e-10*jnp.eye(M)
L = cholesky(S)
z = L.T@y

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()
ax.plot(y, 'x', color=(0.8,0.8,0.8))
ax.plot(z, linewidth=2)
ax.set_xlabel('Dimension: $M$', fontsize=16)
ax.set_ylabel('Random Variable $Y$, $Z$', fontsize=16)
plt.show()
```
??? success "計算結果"
    ![相関のある 1000 次元データ (青色)](./img/random_1000dim_with_correlation.png)

相関のないオリジナルのデータを灰色の散布図で, 相関を与えたデータを青色の線で示しました. このように行列の index が近いデータほど相関が強くなるということを要請すると, なめらかに変化するデータを得ることができました.

上記の例では相関のない多次元正規分布からのデータを経由しましたが, 多くの数値計算ライブラリでは直接相関のあるデータをサンプルするための機能が提供されています. 以下では `numpyro` の `MultivariateNormal` を使用してみます. この関数は `covariance_matrix` というオプションで共分散行列を与えることができます.

``` python
key = random.PRNGKey(42)
w = dist.MultivariateNormal(covariance_matrix=S).sample(key)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()
ax.plot(y, 'x', color=(0.7,0.7,0.7))
ax.plot(z, linewidth=2)
ax.plot(w, linewidth=2)
ax.set_xlabel('Dimension: $M$', fontsize=16)
ax.set_ylabel('Random Variable $Y$, $Z$', fontsize=16)
plt.show()
```
??? success "計算結果"
    ![相関のある 1000 次元データ (オレンジ)](./img/random_1000dim_with_correlation-2.png)


さきほどコレスキー分解で作成したデータ系列と全く同じデータがサンプルされました. おそらく内部で数学的に同値な処理が行われているのだと思います. 与えている `key` を変えると違うデータ系列がサンプルされます. `PRNGKey` の引数を変えて試してみてください.


## カーネルによるデータの表現
上記の例ではデータ系列 $\{Y_i\}$ とその index を使用して共分散行列 $\Sigma_{m,n}$ を作成しました. 結局のところ, この操作はコレスキー分解可能な行列 $\Sigma$ を与えることができれば計算できます. ここではさらに一歩進んで行列 $\Sigma$ を生成するための関数を考えましょう. $x \in \mathbb{R}^m$, $z \in \mathbb{R}^n$ を引数にとって $m{\times}n$ 行列を返す関数 $K$ を考えます.[^3]

[^3]: $x$, $z$ は $x \in \mathbb{R}^{m{\times}d}$, $z \in \mathbb{R}^{n{\times}d}$ のように多次元のデータ系列でも問題ありません. 関数 $K$ の出力が $m{\times}n$ 行列になっていることが重要です.

$$
 K(x, z) \in \mathbb{R}^{m{\times}n}
 \quad \left(
 x \in \mathbb{R}^m,~
 z \in \mathbb{R}^n \right).
$$

データ系列 $\{X_i, Y_i\}_{i=1{\ldots}M}$ に対して以下の行列を計算します.

$$
\Sigma = K(\{X_i\}, \{X_i\}).
$$

$\Sigma$ は $M{\times}M$ 行列になります. $\Sigma$ がデータ系列 $\{Y_i\}$ の共分散行列をうまく近似できるように関数 $K$ を設計することができれば, 関数 $K$ はデータ系列 $\{X_i,Y_i\}_{i=1{\ldots}N}$ のもつ __なめらかさを学習する__  ことができます. この関数 $K$ をカーネルと呼びます.

カーネル $K$ の関数系が決まってしまえば, 任意のデータ系列 $\{\tilde{X}_i\}_{i={\ldots}N}$ に対して以下の行列を計算できます.

$$
\tilde{\Sigma} = K(\{\tilde{X}_i\},\{\tilde{X}_i\})
$$

この行列を共分散行列として多次元正規分布から $\{\tilde{Y}_i\}_{i=1{\ldots}N}$ をサンプルすることができます. 新たに生成したデータ系列 $\{\tilde{X}_i,\tilde{Y}_i\}_{i=1{\ldots}N}$ はもとのデータ系列 $\{X_i,Y_i\}_{i=1{\ldots}M}$ と同等のなめらかさを持ったデータ系列です.

サンプルとしてカーネル関数を以下のように定義した結果を示します. $\{X_i\}_{i=1{\ldots}M}$ は一様分布からサンプルしています. カーネルのパラメタ `length` は $l=3$ としました. 多次元正規分布からデータ系列を 5 つサンプルした結果を図示しています.

$$
K(x,z;l) = \exp\left(-\frac{(x-z)^2}{l^2}\right)
$$

``` python
def kernel(x,z,length,epsilon=1e-10):
  sq2d = jnp.power((x[:,None]-z)/length,2)
  return jnp.exp(-sq2d) + epsilon*jnp.eye(x.size)

key = random.PRNGKey(42)

M = 50
x = jnp.sort(dist.Uniform(0,10).expand([M,]).sample(key))
K = kernel(x,x,length=3.0)

y = dist.MultivariateNormal(covariance_matrix=K).expand([5,]).sample(key)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()
ax.plot(x, y.T, 'o-')
ax.set_xlabel('Explanary Variable: $X$', fontsize=16)
ax.set_ylabel('Random Variable $Y$', fontsize=16)
plt.show()
```

![カーネル関数からサンプリングしたデータ系列](./img/random_kernel_sampling.png)

同じような __なめらかさ__ を持ったデータ系列を 5 つサンプルすることができました. 実際に多次元正規分布からサンプルしているのは $\{\tilde{Y}_i\}_{i=1{\dots}N}$ というデータセットなのですが, 考え方によってはカーネル関数 $K$ によって定義されたある __なめらかさ__ を持った関数の集合 $G_f (K)$ から関数 $f$ をひとつサンプリングして,

$$
 \tilde{Y}_i = f(\tilde{X}_i) \quad f \in G_f(K)
$$

という式で $\tilde{Y}_i$ を計算しているように見えてこないでしょうか. 関数の集合 $G_f(K)$ の中から, 観測されたデータセット $\{X_i, Y_i\}_{i=1{\ldots}M}$ をうまく表現できる関数だけを抜き出すことを考えてみましょう. 抜き出した関数をうまくつかえばデータの補間や予測に活用できそうです. また, 抜き出した関数のばらつきを調べることで, 補間した値の信頼性を議論することもできそうです. 以下の章では実際にガウス過程を用いてデータを解析していきます.