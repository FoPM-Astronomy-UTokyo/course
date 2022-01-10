# さまざまなカーネル関数

この章ではガウス過程で広く使われているカーネルとその性質を紹介します. まずは使用するモジュールを import します.

``` python
import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist

rng_key = random.PRNGKey(42)
```

また, カーネルを図示するための関数 `display_kernel` を定義します. この関数は 2 つのパネルを表示します. この章で紹介するカーネル関数 $K(x,z)$ はすべて $d = x{-}z$ の関数として定義されています. そこで上のパネルではカーネルの値を $d$ の関数として表示しています. 下のパネルではここで定義したカーネルからサンプリングしたデータ系列を 5 つ表示します.

``` python
def display_kernel(key, kernel, N=5, **options):
    fig,axes = plt.subplots(2,1,
        figsize=(16,9), gridspec_kw={'height_ratios': [3, 5]})

    z = jnp.linspace(-10,10,801)
    axes[0].plot(z,kernel(jnp.array([0]),z,**options).T)
    axes[0].set_xlabel('Distance $d$', fontsize=16)
    axes[0].set_ylabel('Correlation', fontsize=16)

    keys = random.split(key, 5)
    x = jnp.linspace(0,10,200)
    for key in keys:
        y = dist.MultivariateNormal(
            loc = jnp.zeros(x.shape[0]),
            covariance_matrix=kernel(x,x,**options)
        ).sample(key)
        axes[1].plot(x, y)
    axes[1].set_xlabel('Explainatory Variable $x$', fontsize=16)
    axes[1].set_ylabel('Explained Variable $y$', fontsize=16)
    plt.show()
```

## Noise Kernel
最初に相関のないノイズを表すカーネルを紹介します. ノイズには相関がないので共分散行列は単位行列で与えられます. 上のパネルに着目すると $d=0$ のときのみ値を持つ関数になっていることがわかります. ノイズの大きさを表すパラメタ `noise` を持ちます.

以下のサンプルコードでは図示の都合で $\delta(X_i{-}Z_j)$ として定義していますが, 実用的は単位行列に `noise` をかけたものを使用することになります.

``` python
def noise_kernel(X, Z, noise):
    return noise*(jnp.abs(X[:,None]-Z)==0)

display_kernel(rng_key, noise_kernel, noise=1.0)
```
![Noise Kernel](./img/kernel_noise.png)



## Radial Basis Function Kernel

Radial Basis Function Kernel (RBF Kernel) は以下のように定義される関数です.[^1]

[^1]: Squared Exponential Kernel や Exponentiated Quadratic Kernel などと呼ばれることもあります.

$$
K(x,z) = \sigma^2 \exp\left(-\frac{(x-z)^2}{2l^2}\right).
$$

$\sigma^2$ は変動の大きさを表すパラメタです. また $l$ は相関長 (なめらかさ) を表すパラメタです. 以下にサンプルコードを示します.[^2]

[^2]: 計算の安定化のために `jitter` 成分を加えています.

``` python
def rbf_kernel(X, Z, var, length, jitter=1e-10):
    dXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    K = var * jnp.exp(-0.5*dXsq)
    return K + jitter*jnp.eye(X.size)

display_kernel(rng_key, rbf_kernel, var=0.3, length=1.0)
```
![RBF Kernel](./img/kernel_radialbasis.png)

上のパネルを見ると, このカーネルは式が示すとおり Gaussian の形状をしていることがわかります. この図で裾野が広いほど離れたデータ点との相関が強くなるため, よりなめらかな関数になるという傾向にあります. 相関長 $l$ (`length`) の値を調整することでさまざまななめらかさをもった関数を定義できます.


## Matérn Kernels

Matérn Kernel は以下の式で定義される関数です.

$$
K(x,z) = C_\nu(d) =
\sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
\left(\sqrt{2\nu}\frac{d}{l}\right)^\nu
K_\nu\left(\sqrt{2\nu}\frac{d}{l}\right).
$$

ただし $d = |x{-}z|$ で定義されます. $\Gamma(x)$ はガンマ関数です. また $K_\nu(x)$ は第 2 種の変形ベッセル関数です. $\sigma^2$ は変動の大きさを表すパラメタです. また, $l$ は相関長を表すパラメタです.

Matérn Kernel から生成されたデータ系列は $\lceil\nu\rceil-1$ 回微分可能という特徴を持ちます. Matérn Kernel は $\nu = 1/2,\, 3/2,\, 5/2$ のケースがよく使用されます. 以下にそれぞれのケースを示します.

### Matérn 1/2 Kernel
$\nu = 1/2$ の場合, カーネル関数は以下の式で与えられます.

$$
K(x,z) = \sigma^2 \exp\left( - \frac{d}{l} \right).
$$

サンプルコードは以下のとおりです.

``` python
def matern12_kernel(X, Z, var, length, jitter=1e-10):
    dX = jnp.abs((X[:, None] - Z) / length)
    k = var * jnp.exp(-dX)
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, matern12_kernel, var=0.3, length=1.0)
```
![Matérn 1/2 Kernel](img/kernel_matern12.png)

このカーネルは $d=0$ 付近で変化が急激であり, 不連続な形状をしています. そのため, 隣接するデータ間の相関が弱く, 生成されたデータ系列もギザギザした形状 (微分不可能) をしています. しかし, 大局的には相関があるデータ系列となっており, ランダムウォークに近い見た目をしています.


### Matérn 3/2 Kernel
$\nu = 3/2$ の場合, カーネル関数は以下の式で与えられます.

$$
K(x,z) = \sigma^2\left(1 + \sqrt{3}\frac{d}{l}\right)
\exp\left(-\sqrt{3}\frac{d}{l}\right).
$$

サンプルコードは以下のとおりです.

``` python
def matern32_kernel(X, Z, var, length, jitter=1e-10):
    dX = jnp.abs((X[:, None] - Z) / length)
    k = var * (1+jnp.sqrt(3)*dX)*jnp.exp(-jnp.sqrt(3)*dX)
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, matern32_kernel, var=0.3, length=1.0)
```
![Matérn 3/2 Kernel](./img/kernel_matern32.png)

$\nu = 3/2$ の場合は $d=0$ 付近の形状が少しなめらかになりました. その影響もあってか生成されたデータ系列も角がとれて丸みを帯びています. $\lceil\nu\rceil{-}1 = 1$ なので 1 回微分可能な見た目をしている (はず) です.


### Matérn 5/2 Kernel
$\nu = 5/2$ の場合, カーネル関数は以下の式で与えられます.

$$
K(x,z) = \sigma^2\left(1 + \sqrt{5}\frac{d}{l} + \frac{5d^2}{3l^2}\right)
\exp\left(-\sqrt{5}\frac{d}{l}\right).
$$

サンプルコードは以下のとおりです.
``` python
def matern52_kernel(X, Z, var, length, jitter=1e-10):
    dX = jnp.abs((X[:, None] - Z)/length)
    k = var*(1+jnp.sqrt(5)*dX+5/3*jnp.power(dX,2))*jnp.exp(-jnp.sqrt(5)*dX)
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, matern52_kernel, var=0.3, length=1.0)
```
![Matérn 5/2 Kernel](./img/kernel_matern52.png)

$\nu = 5/2$ のケースでは関数のプロファイルがさらに丸みを帯びてきました. 上のパネルの形状もかなり Gaussian に近づいてきました. 一方で, 生成されたデータ系列を見ると RBF Kernel に比べて少しガタツキが目立つように感じられます. このデータ系列は 2 回までは微分可能です.

## Rational Quadratic Kernel

Rational Quadratic Kernel (RQ Kernel) は以下の関数で定義されます.

$$
K(x,z) = \sigma^2 \left(1 + \frac{(x-z)^2}{2{\alpha}l^2}\right)^{-\alpha}.
$$

このカーネルは異なる `length` を持った RBF Kernel の重ね合わせによって作ることができます. $\alpha \to \infty$ の極限で RBF Kernel と一致します. 有限の $\alpha$ に対しては, RBF Kernel よりも少し裾野の広いカーネルとして働きます.

サンプルコードは以下のとおりです.

``` python
def rq_kernel(X, Z, var, alpha, length, jitter=1e-10):
    dXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.power(1+dXsq/2/alpha, -alpha)
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, rq_kernel, var=0.3, alpha=1.0, length=1.0)
```
![Rational Quadratic Kernel](./img/kernel_rationalquadratic.png)

今回は $\alpha = 1$ を採用しました. RBF Kernel に比べてかなり裾野が広い分布になっています. 長いスパンでの相関が強くなっているはずですが, 生成されたデータ系列からは RBF Kernel との違いを読み取ることは難しいかもしれません.

??? tip "RQ Kernel の求めかた"
    RBF Kernel を以下のようにガンマ分布で重み付けをして足し合わせた関数を考えます.

    $$
    K(x,z) = \int^\infty_0 \exp\left(-z\xi\frac{(x-z)^2}{2l^2}\right)
    \Gamma(\xi; \alpha, \alpha^{-1})\,\mathrm{d}\xi
    $$

    ここで $\Gamma(\xi; \alpha, \alpha^{-1})$ は shape parameter が $\alpha$ で平均が $\alpha\alpha^{-1}=1$ のガンマ分布関数です. 定義式から実際に積分計算を進めてみると RQ Kernel と一致します.

    $$
    \begin{aligned}
    K(x,z) &=
      \frac{\alpha^\alpha}{\Gamma(\alpha)}
      \int^\infty_0
      \exp\left(-\left(\frac{(x-z)^2}{2l^2}{+}\alpha\right)\xi\right)
      \xi^{\alpha-1}\,\mathrm{d}\xi \\
      &= \frac{\alpha^\alpha}{\Gamma(\alpha)}
      \left(\frac{(x-z)^2}{2l^2}{+}\alpha\right)^{-\alpha}
      \int^\infty_0 \exp(-\eta)\eta^{\alpha-1}\,\mathrm{d}\eta \\
      &= \frac{\alpha^\alpha}{\Gamma(\alpha)}
      \left(\frac{(x-z)^2}{2l^2}{+}\alpha\right)^{-\alpha}\Gamma(\alpha) \\
      &= \left(1 + \frac{(x-z)^2}{2\alpha l^2}\right)^{-\alpha}
    \end{aligned}
    $$

    計算式中の $\Gamma(x)$ はガンマ関数です. ガンマ分布関数 $\Gamma(\xi; \alpha, \alpha^{-1})$ は $\alpha \to \infty$ の極限でデルタ関数 $\delta(\xi-1)$ に収束します. 同じ極限操作で RBF Kernel に一致したことに納得がいくのではないでしょうか.


## Periodic Kernel
Periodic Kernel は以下の関数で定義されます.

$$
K(x,z) = \sigma^2\exp\left(
  -\frac{2}{l^2}\sin^2\left(\frac{\pi|x-z|}{P}\right)
  \right)
$$

$\sigma^2$ は変動の大きさを, $l$ は相関長を, $P$ は周期を表すパラメタです.

サンプルコードは以下のとおりです.

``` python
def periodic_kernel(X, Z, var, length, period, jitter=1e-10):
    pX = jnp.pi*jnp.abs((X[:, None] - Z))/period
    k = var * jnp.exp(-2*jnp.power(jnp.sin(pX)/length,2))
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, periodic_kernel, var=0.3, length=1.0, period=3.0)
```
![Periodic Kernel](./img/kernel_periodic.png)

これまでのカーネルは $d \sim 0$ 付近でのみ大きい値を持つ関数でしたが, Periodic Kernel はその名の通り周期的に相関のピークが現れます. 今回は $P=3$ を採用したので, $\Delta{d} = 3$ でピークが現れていることがわかります. この構造は生成されたデータ系列にも反映されており, データに明確な周期構造を確認できます.

## Bias Kernel
Bias Kernel は以下の関数で定義されます.

$$
K(x,z) = \sigma_b^2.
$$

すべての要素が同じ定数になります. どれだけ距離が離れていても相関の強さが変わらないため, 定数値になります. $b$ は定数値がどれだけ 0 から離れるかをコントロールします.

サンプルコードは以下のとおりです.

``` python
def bias_kernel(X, Z, bias, jitter=1e-10):
    nx,nz = X.size,Z.size
    k = bias*jnp.ones([nx,nz])
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, bias_kernel, bias=1.0)
```
![Bias Kernel](./img/kernel_bias.png)

カーネルの値も生成されたデータ系列も一定値になります.


## Linear Kernel
Linear Kernel は以下の関数で定義されます.

$$
K(x,z) = \sigma^2(x-c)(z-c).
$$

$\sigma^2$ は変動の大きさを表すパラメタです. また, $c$ はゼロ点の位置を示すアンカーポイントを表しています. サンプルコードは以下のとおりです.

``` python
def linear_kernel(X, Z, var, anchor, jitter=1e-10):
    k = var*(X[:, None] - anchor)*(Z - anchor)
    return k + jitter*jnp.eye(X.size)

display_kernel(rng_key, linear_kernel, var=1.0, anchor=5.0)
```
![Linear Kernel](./img/kernel_linear.png)

$x=c$ の位置では分散が 0 になるため生成されたデータ系列も 0 になります. このカーネルは負の値もとりうるため, もはや共分散という考え方では解釈できないものになっています.


## Mixed Kernels
紹介してきたカーネルは組み合わせて使うこともできます. データ系列に期待される性質から適切なカーネルを選択して組み合わせます. カーネルの選択だけでなく, 組み合わせ方に依っても異なるデータ系列が生成されます. 以下に簡単なサンプルを示します.

以下のサンプルコードは RBF Kernel と Periodic Kernel の和で定義したカーネルです.

``` python
mixed_kernel = lambda x,z,var1,var2,len1,len2,period: \
    rbf_kernel(x,z,var1,len1)+periodic_kernel(x,z,var2,len2,period)

display_kernel(rng_key, mixed_kernel,
    var1=1.0, var2=1.0, len1=2.0, len2=1.0, period=1.0)
```
![Mixed Kernel (RBF + Periodic)](./img/kernel_mixed_movingperiodic.png)

$d \sim 0$ 付近で相関が高くなる構造と, 周期的に相関が高くなる構造を併せ持ったカーネルになっています. 実際に生成されたデータ系列を見てみると, 緩やかに変動するベースラインの上に周期的なシグナルが乗っていることがわかります.


以下のサンプルコードは RBF Kernel と Periodic Kernel の積で定義したカーネルです.

``` python
mixed_kernel = lambda x,z,var,length,period: \
    rbf_kernel(x,z,var,length)*periodic_kernel(x,z,1.0,period,length)

display_kernel(rng_key, mixed_kernel, var=1.0, length=2.0, period=1.0)
```
![Mixed Kernel (RBF &times; Periodic)](./img/kernel_mixed_localperiodic.png)

Periodic Kernel の周期的なピーク構造が急激に減衰するようになりました. 生成されたデータ系列を確認してみると, 短いスパンでは周期的な構造がはっきり見えています. しかし, 周期的なシグナルの形状は徐々に崩れていき, $x \sim 1$ 付近と $x \sim 9$ 付近では違う形状になっています (ただし変動の周期は同じです). このカーネルは周期こそ不変ですが, 波形が徐々に変化するシグナルを生成できます.[^3]

[^3]: Locally Periodic Kernel などの名前で呼ばれているようです ([Kernel Cookbook][cookbook]).


## 参考資料
- [Kernel Cookbook][cookbook]
- [Gaussian processes (3/3) - exploring kernels][kernels]

[cookbook]: https://www.cs.toronto.edu/~duvenaud/cookbook/
[kernels]: https://peterroelants.github.io/posts/gaussian-process-kernels/