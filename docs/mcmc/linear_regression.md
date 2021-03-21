# 演習問題: 線形回帰

ここでは最もシンプルな線形回帰問題にたいして MCMC を使用してみます. 題材として使用するのは銀河の中心にある超大質量ブラックホール (supermassive blackhole) と, 銀河バルジ (bulge) の速度分散の関係です.

## データ
[Harris et al. (2013)][Harris2013] は銀河における球状星団の性質を調べるために, 先行研究によって測定されたさまざまな銀河の物理量をまとめた[カタログ][H2013]を作成しました. 今回はそのなかから銀河バルジの速度分散 (主に可視光の分光観測から推定) と銀河中心の超大質量ブラックホールの質量 (X 線光度などから推定) の値を使用します.

演習用に整形したデータを以下からダウンロードしてください.

 ファイル名 | 形式
 ---------- | ----------
 [exercise_linear_regression.csv][data] | csv

データテーブルには以下のカラムが含まれています.

 カラム名  | 説明
 --------  | ----------
 galaxy    | 銀河名/カタログ ID[^1]
 ra        | 赤経座標 (J2000)[^1]
 dec       | 赤緯座標 (J2000)[^1]
 dist      | 距離 (Mpc)[^1]
 dist_err  | 距離の不定性 (Mpc)[^1]
 logsig    | 銀河バルジの速度分散 (対数) $\log_{10}\sigma_e$ (km/s)
 logsig_err| 銀河バルジの速度分散の不定性 $\varepsilon_\sigma$
 logM_B    | 超大質量ブラックホールの質量 (対数) $\log_{10}{M_B}$ (${M_\odot}$)
 logM_B_err| 超大質量ブラックホールの質量の不定性 $\varepsilon_M$

[^1]: 今回は使いません.


まずはデータの関係を図示します.

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

table = pd.read_csv('./exercise_linear_regression.csv')
print(table)

fig = plt.figure()
ax = fig.add_subplot()
ax.errorbar(
  x = table.logsig, y = table.logM_B,
  xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
fig.tight_layout()
plt.show()
```

??? summary "計算結果"
    ```
          galaxy         ra        dec  ...  logsig_err  logM_B  logM_B_err
    0   MilkyWay   0.000000   0.000000  ...       0.076    6.61       0.040
    1     NGC221   0.711607  40.865165  ...       0.017    6.46       0.090
    2     NGC224   0.712316  41.268883  ...       0.021    8.18       0.150
    3     NGC253   0.792531 -25.288442  ...       0.073    7.00       0.300
    4     NGC821   2.139199  10.995008  ...       0.020    8.23       0.205
    5    NGC1023   2.673329  39.063253  ...       0.020    7.64       0.040
    6    NGC1316   3.378256 -37.208211  ...       0.006    8.24       0.200
    7    NGC1332   3.438111 -21.335276  ...       0.018    9.17       0.060
    8    NGC1399   3.641408 -35.450626  ...       0.020    8.69       0.065
    9    NGC2778   9.206775  35.027417  ...       0.019    7.15       0.300
    10   NGC3031   9.925869  69.065438  ...       0.021    7.84       0.085
    11   NGC3115  10.087214  -7.718556  ...       0.020    8.96       0.170
    12   NGC3377  10.795104  13.985641  ...       0.020    8.25       0.240
    13   NGC3379  10.797108  12.581611  ...       0.021    8.04       0.240
    14   NGC3384  10.804699  12.629401  ...       0.021    7.04       0.210
    15   NGC3414  10.854492  27.974833  ...       0.014    8.40       0.070
    16   NGC3585  11.221418 -26.754864  ...       0.020    8.51       0.140
    17   NGC3607  11.281816  18.051899  ...       0.020    8.15       0.125
    18   NGC3608  11.283036  18.148538  ...       0.021    8.67       0.095
    19   NGC3842  11.733936  19.949696  ...       0.011    9.99       0.125
    20   NGC4261  12.323060   5.825041  ...       0.020    8.71       0.085
    21   NGC4291  12.338247  75.370944  ...       0.021    8.98       0.140
    22   NGC4350  12.399394  16.693471  ...       0.017    8.74       0.100
    23   NGC4374  12.417685  12.887071  ...       0.020    8.96       0.045
    24   NGC4459  12.483339  13.978556  ...       0.020    7.84       0.080
    25   NGC4472  12.496331   8.000389  ...       0.004    9.26       0.150
    26   NGC4473  12.496907  13.429397  ...       0.020    7.95       0.240
    27   NGC4486  12.513724  12.391217  ...       0.020    9.77       0.030
    28  NGC4486A  12.516033  12.270333  ...       0.019    7.15       0.120
    29   NGC4552  12.594402  12.556115  ...       0.006    8.68       0.045
    30   NGC4564  12.607493  11.439400  ...       0.021    7.92       0.120
    31   NGC4594  12.666513 -11.623010  ...       0.021    8.72       0.435
    32   NGC4621  12.700637  11.647308  ...       0.006    8.60       0.065
    33   NGC4649  12.727789  11.552672  ...       0.021    9.63       0.100
    34   NGC4697  12.809995  -5.800602  ...       0.019    8.28       0.090
    35   NGC4889  13.002237  27.977031  ...       0.006   10.32       0.435
    36   NGC5128  13.424479 -43.018118  ...       0.020    7.71       0.160
    37    IC4296  13.610847 -33.965822  ...       0.021    9.13       0.065
    38   NGC5813  15.019805   1.702009  ...       0.006    8.84       0.070
    39   NGC5845  15.100215   1.633972  ...       0.020    8.69       0.140
    40   NGC5846  15.108124   1.606291  ...       0.008    9.04       0.080
    41   NGC6086  16.209883  29.484478  ...       0.012    9.56       0.160
    42   NGC7332  22.623476  23.798260  ...       0.011    7.11       0.190
    43    IC1459  22.952945 -36.462176  ...       0.011    9.45       0.195
    44   NGC7457  23.016647  30.144889  ...       0.019    7.00       0.300
    45   NGC7768  23.849610  27.147336  ...       0.021    9.11       0.150

    [46 rows x 9 columns]
    ```
    ![データの関係](img/execrcise_linear_regression_quick_view.png)


## 回帰直線の導出


### Y 軸の不定性を考慮した回帰
おおよそ線形の関係で近似できそうなことが分かりました. ここでは Harris et al. (2013) を参考にして以下の式を仮定します. ただし $\sigma_0$ は 200 km/s とします.

$$
\log_{10} \frac{M_B}{M_\odot}
= \alpha + \beta \log_{10}\frac{\sigma_e}{\sigma_0}.
$$

誤差は正規分布で近似できるという単純なモデルを採用します. まずは Y 軸 ($\log{M_B}$) の不定性だけを考えて尤度関数 (確率) を導出します.[^2]

$$
\log{L(\alpha,\beta; D)}
= -\sum_{i=1}^n
\frac{
  \left( \log_{10}{M_{B,i}} - \alpha - \beta\log_{10}\sigma_{e,i} \right)^2
}{2{\varepsilon_{M,i}}^2}.
$$

この尤度関数に対して MCMC を用いて $\alpha$, $\beta$ の分布を求めてください. また $\alpha$, $\beta$ の平均値を polynomial fitting (`numpy.polyfit`) の結果と比較してください.

[^2]: これは尤度関数なので事前分布を仮定して掛けないと正しく確率としては扱えませんが, ここでは一様な無情報事前分布を仮定したとして, このまま確率として計算をしてしまいます.


??? note "Example"
    ``` python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mhmcmc import MHMCMCSampler, GaussianStep

    table = pd.read_csv('./exercise_linear_regression.csv')

    def log_likelihood(x):
      delta = table.logM_B - (x[0] + x[1]*(table.logsig-np.log10(200)))
      sigma = table.logM_B_err
      return -np.sum(delta**2/sigma**2/2)

    step = GaussianStep(np.array([0.02, 0.15]))
    model = MHMCMCSampler(log_likelihood, step)
    x0 = np.array([8.0, 5.0])
    model.initialize(x0)

    sample = model.generate(51000)
    sample = sample[1000:]

    x = np.linspace(-0.5,0.5,50)
    a,b = sample.mean(axis=0)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    for _a,_b in sample[::1000,:]:
      ax.plot(x, _a+_b*x, color='orange', alpha=0.1)
    ax.errorbar(
      x = table.logsig-np.log10(200), y = table.logM_B,
      xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
    ax.plot(x, a+b*x)
    ax.set_xlabel('$\log\sigma_e$ (km/s)')
    ax.set_ylabel('$\log M_B$ ($M_\odot$)')
    fig.tight_layout()
    plt.show()

    p = np.polyfit(
      table.logsig-np.log10(200), table.logM_B, 1,
      w = 1./table.logM_B_err)

    print(f'MCMC inference: alpha={a:.3f}, beta={b:.3f}')
    print(f'polyfit result: alpha={p[1]:.3f}, beta={p[0]:.3f}')
    ```

??? summary "計算結果"
    ```
    MCMC inference: alpha=8.195, beta=5.001
    polyfit result: alpha=8.195, beta=5.002
    ```
    ![データと回帰直線](img/execrcise_linear_regression_yerror.png)

    参考までに上記のサンプルで出力したトレースと自己相関関数を示します.
    ![トレース](img/execrcise_linear_regression_yerror_trace.png)
    ![自己相関関数](img/execrcise_linear_regression_yerror_autocorr.png)



### X,Y 軸の不定性を考慮した回帰

データには速度分散の不定性も与えられていました. 尤度関数を変更して速度分散の不定性も考慮に入れてください.

速度分散が $\Delta$ だけ変わると縦軸は $\beta\Delta$ だけ変動します. よって速度分散の不定性を考慮に入れると尤度関数は以下のように書けます.

$$
\begin{aligned}
\log{L(\alpha,\beta; D)} = -\sum_{i=1}^n
\left[ \frac{\Delta_i^2}{2S_i^2} + \log{S_i}\right],\qquad\qquad &\\
\left\{~\begin{aligned}
\Delta_i &= \log_{10}{M_{B,i}} - \alpha - \beta\log_{10}\sigma_{e,i}, \\
S_i      &= {\varepsilon_{M,i}}^2 + \beta^2{\varepsilon_{\sigma,i}}^2.
\end{aligned}\right.&
\end{aligned}
$$

この尤度関数に対して MCMC を用いて $\alpha$, $\beta$ の分布を求めてください.


??? note "Example"
    ``` python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mhmcmc import MHMCMCSampler, GaussianStep

    table = pd.read_csv('../../data/mcmc/exercise_linear_regression.csv')

    def log_likelihood(x):
      delta = table.logM_B - (x[0] + x[1]*(table.logsig-np.log10(200)))
      sqsig = table.logM_B_err**2 + x[1]**2*table.logsig_err**2
      return -np.sum(delta**2/sqsig/2)

    step = GaussianStep(np.array([0.02, 0.15]))
    model = MHMCMCSampler(log_likelihood, step)
    x0 = np.array([8.0, 5.0])
    model.initialize(x0)

    sample = model.generate(51000)
    sample = sample[1000:]

    x = np.linspace(-0.5,0.5,50)
    a,b = sample.mean(axis=0)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    for _a,_b in sample[::1000,:]:
      ax.plot(x, _a+_b*x, color='orange', alpha=0.1)
    ax.errorbar(
      x = table.logsig-np.log10(200), y = table.logM_B,
      xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
    ax.plot(x, a+b*x)
    ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
    ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
    fig.tight_layout()
    fig.savefig('execrcise_linear_regression_xyerror.png')
    plt.show()
    ```

??? summary "計算結果"
    ```
    MCMC inference: alpha=8.300, beta=4.919
    ```
    ![データと回帰直線](img/execrcise_linear_regression_xyerror.png)

    参考までに上記のサンプルで出力したトレースと自己相関関数を示します.
    ![トレース](img/execrcise_linear_regression_xyerror_trace.png)
    ![自己相関関数](img/execrcise_linear_regression_xyerror_autocorr.png)


### モデルの不定性を考慮した回帰

{++予備課題++} これまでの計算で $\alpha$, $\beta$ の確率分布を求めましたが, 散布図に重ねてプロットをしてみると, 測定誤差では説明ができないほど傾向から外れた点も多いことが分かります. ひとつの可能性は測定誤差を過小評価しているということです. また別の可能性として, ここでは考慮できていない物理量があるため, 直線からのバラつきとして intrinsic scatter が現れていると見ることもできます.

ここでは intrinsic scatter の大きさを見積もるために変数をひとつ増やして計算をしてみます. 線形回帰で得られる値に一様に不定性 $\varepsilon$ を加えて尤度関数を以下のように定義します.

$$
\begin{aligned}
\log{L(\alpha,\beta; D)} = -\sum_{i=1}^n
\left[ \frac{\Delta_i^2}{2S_i^2} + \log{S_i}\right]
+ \log{\operatorname{Gamma}(\tau, k, \theta)},\qquad &\\
\left\{~\begin{aligned}
\Delta_i &= \log_{10}{M_{B,i}} - \alpha - \beta\log_{10}\sigma_{e,i}, \\
S_i      &= {\varepsilon_{M,i}}^2 + \beta^2{\varepsilon_{\sigma,i}}^2 + \varepsilon^2, \\
\tau     &= \varepsilon^{-2}.
\end{aligned}\right.&
\end{aligned}
$$

$\tau$ が負値を取らないように事前分布として Gamma 分布を仮定しました. 以下のサンプルでは無情報であることを意味するため $k$, $\theta$ にそれぞれ $10^{-3}$, $10^3$ を与えています.[^3]

[^3]: このとき $\tau$ が大きくない範囲では ${\operatorname{Gamma}(\tau,k,\theta)} \sim k\tau^{-1}$ と近似でき, Jefferys の無情報事前分布と一致します. Gamma 分布はパラメタの定義の仕方に $(k,\theta)$ と $(\alpha,\beta)$ の 2 通りあるので気をつけて使ってください.


??? note "Example"
    ``` python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mhmcmc import MHMCMCSampler, GaussianStep

    table = pd.read_csv('../../data/mcmc/exercise_linear_regression.csv')

    def log_gamma(x,k,t):
      return (k-1)*np.log(x)-x/t if x>0 else -1e10

    def log_likelihood(x):
      delta = table.logM_B - (x[0] + x[1]*(table.logsig-np.log10(200)))
      sqsig = table.logM_B_err**2 + x[1]**2*table.logsig_err**2 + x[2]
      return -np.sum(delta**2/sqsig/2+np.log(sqsig)/2) \
        + log_gamma(1/x[2],1e-3,1e3) if (sqsig > 0).all() else -1e10

    step = GaussianStep(np.array([0.02, 0.15, 0.03]))
    model = MHMCMCSampler(log_likelihood, step)
    x0 = np.array([8.0, 5.0, 0.5])
    model.initialize(x0)

    sample = model.generate(51000)
    sample = sample[1000:]
    sample[:,2] = np.sqrt(sample[:,2])
    x = np.linspace(-0.5,0.5,50)
    a,b,e = sample.mean(axis=0)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    ax.fill_between(x, a+b*x-3*e, a+b*x+3*e, color='gray', alpha=0.05)
    ax.fill_between(x, a+b*x-e, a+b*x+e, color='gray', alpha=0.10)
    for _a,_b,_e in sample[::1000,:]:
      ax.plot(x, _a+_b*x, color='orange', alpha=0.05)
    ax.errorbar(
      x = table.logsig-np.log10(200), y = table.logM_B,
      xerr = table.logsig_err, yerr = table.logM_B_err, fmt='.')
    ax.plot(x, a+b*x)
    ax.set_xlabel('$\log_{10}\sigma_e$ (km/s)')
    ax.set_ylabel('$\log_{10}M_B$ ($M_\odot$)')
    fig.tight_layout()
    plt.show()

    print(f'MCMC inference: alpha={a:.3f}, beta={b:.3f}, epsilon={e:.3f}')
    ```

??? summary "計算結果"
    ```
    MCMC inference: alpha=8.332, beta=4.430, epsilon=0.366
    ```
    1-&sigma;, 3-&sigma; の不定性をグレーの領域で表しています.
    ![データと回帰直線](img/execrcise_linear_regression_epsilon.png)

    参考までに上記のサンプルで出力したトレースと自己相関関数を示します.
    ![トレース](img/execrcise_linear_regression_epsilon_trace.png)
    ![自己相関関数](img/execrcise_linear_regression_epsilon_autocorr.png)


[Harris2013]: https://doi.org/10.1088/0004-637X/772/2/82
[H2013]: https://www.physics.mcmaster.ca/~harris/GCS_table.txt
[data]: https://raw.githubusercontent.com/FoPM-Astronomy-UTokyo/course/main/data/mcmc/exercise_linear_regression.csv
