---
layout: post
title:  "Conditional Accuracy"
date:   2020-10-24
blurb: "Problems with Tests for Conditional Expected Accuracy."
---


Selecting the model that performs best on metric on a test set is standard practice. Tests for equal expected accuracy seek to determine whether the observed loss difference is due to chance. 
If the results of this interesting preprint, _Can Two Forecasts Have the Same Conditional Expected Accuracy?_ (by Yinchu Zhu and Allan Timmermann, from now on: ZT), available [here](https://arxiv.org/abs/2006.03238), hold up, a common approach for testing equal expected accuracy, the Giacomini & White (2006; GW) test, should not be used in practice. I will briefly line out the background and the paper here, before verifying their claims using simulated data.

## Summary

### Predictive Accuracy

We have two forecasts for $y_{t+1}$, $f_{1,t}$ and $f_{2,t}$. The loss differential is
$$
\Delta L_{t+1}=L(y_{t+1},f_{1,t})-L(y_{t+1},f_{2,t}),
$$
where $L$ is the loss function, e.g., $L(y,f)=(y-f)^2$. The hypothesis that forecasts  
$$
H_0: \mathbb{E}[\Delta L_{t+1}]=0.
$$
A natural statistic to test for equal expected loss is a t-statistic over a holdout sample,
$$
t = \frac{\Delta \bar{L}_{t+1}}{\hat{\sigma}},
$$
where $\Delta \bar{L}_{t+1}$ is an average over the holdout sample and $\hat{\sigma}$ is a standard error. The task is now to find a set of assumptions and a procedure to estimate $\hat{\sigma}$ under which a distribution of $t$ is easy to analyze. In the original test by Diebold and Mariano (1995; DM) test,  $\hat{\sigma}$ uses a HAC (heteroscedasticity and autocorrelation consistent) estimator to account for potential serial correlation in $\Delta L_{t+1}$.

### The GW test

The null hypothesis of the GW test is
$$
H_0^{cond}: \mathbb{E}[\Delta L_{t+1}|\mathcal{F}_t]=0,
$$
or, that $\Delta L_{t+1}$ is a Martingale Difference Sequence (MDS). $H_0^{cond}$ is stronger than $H_0$: it implies $H_0$, because $\mathbb{E}[\Delta L_{t+1}]=\mathbb{E}[\mathbb{E}[\Delta L_{t+1}|\mathcal{F}_t]]$, but not vice versa. Under $H_0^{cond}$, a valid standard error of $t$ is particularly simple,
$$
\hat{\sigma}= \sqrt{\tfrac{1}{T-m}\sum_{t=m}^{T}(\Delta L_{t+1})^2},
$$
because there is no serial correlation in $\Delta L_{t+1}$. Another benefit of $H_0^{cond}$ is that it implies $\mathbb{E}[h_t\Delta L_{t+1}] = 0$ for any $h_t$-measurable functions, implying additional conditions that can be tested.

It is important to add that GW focus on rolling window models, where $\Delta L_{t+1}$ is based on models that were estimated with the same number of observations. This is because an increasing number of observations, as in an extending window approach, complicates things: more complex models will tend to fare better when there are more observations available. Equal expected accuracy at each point in the holdout sample as implied by, therefore, cannot hold with an expanding window, at least when comparing models with different numbers of parameters.

### The Zhu-Timmermann result

What ZT show is that $H_0^{cond}$ cannot hold in a large number of conditions:

1. When $L(y,f)=(y-f)^2$, $H_0^{cond}$ holds only if forecasts and outcome are related in a unnatural way.
2. Under a general loss function, $H_0^{cond}$ cannot hold if the data generating process does not have finite dependence, i.e., if observations depend on all observations in the past (this common in time series models).
3. Even under finite dependence and under a general loss function, $H_0^{cond}$ cannot hold in two leading examples.

These cases are so common that there appears little hope in salvaging $H_0^{cond}$. ZT, therefore, suggest testing $H_0$ instead, i.e., accounting for serial correlation in the estimation of $\hat{\sigma}$. Ironically, this implies using the original DM test. They also suggest an alternative subsample t-test that appears to fare much better in simulations than the t test with HAC standard error.

The subsample t test is taken from Ibramigov and Müller (2010). The idea is to split the data into $K$ blocks and compare sample means between blocks. The bigger the blocks, the less serially correlated the block means. Denote with $\Delta \bar{L}^{(k)}$ the mean loss difference within the $k$'th block and with $\overline{\Delta L}$ the average $\Delta \bar{L}^{(k)}$. The statistic of the subsample t-test is
$$
S_k = \frac{\sqrt{k}\;\overline{\Delta L}}{\sqrt{\frac{1}{K-1} \sum_k (\Delta \bar{L}^{(k)} - \overline{\Delta L})^2}},
$$
which can be compared to critical of the t-distribution with $K-1$ degrees of freedom.


## Simulation

The purpose of the simulation is to determine whether rejection frequencies are in line with the theoretical distribution under the null hypothesis. The difficulty is in enforcing a comparison where the null hypothesis $H_0$ holds (as we have seen, $H_0^{cond}$ will not hold). The simplest setting is probably comparing a single-parameter model for the mean, with a misspecified model without parameters, under squared error loss. This is the setting used in the paper by ZT, and the one that I adopt here. I, however, use a different distribution.

The outcome variables can take either the values $0$ or $1$,
$$
y_t \sim i.i.d.Bernoulli(p).
$$
Model 1 just forecasts $f_{1,t} = 0$ for all $t$. Its expected loss is $p^2$. Model 2 forecasts 
$$
f_{2,t} = \frac{1}{m}\sum_{s=t-m+1}^{t}y_s.
$$
Model 2's asymptotic expected loss is, from the variance of the limit distribution,
$$
\frac{p(1-p)}{m}.
$$
For a given $m$, we can, therefore choose $p$ to yield equal expected loss for the forecasts:
$$
p = \frac{1}{m+1}.
$$



```python
%load_ext lab_black
```


```python
import warnings
from itertools import product

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
```


```python
def get_p(m):
    return 1 / (m + 1)
```

First, we plot the $p$ that yields equal expected loss for different window sizes $m$.


```python
m = range(1, 100)
p = [get_p(mm) for mm in m]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(m, p)
```




    [<matplotlib.lines.Line2D at 0x1bd68d76208>]




    
![png](conditional_accuracy_files/conditional_accuracy_6_1.png)
    


#### Function definitions

Next, define the functions that generates the rolling mean forecast and the one that computes the loss difference between the two forecasts:


```python
def rolling_forecast(y, m):
    f = []
    for s in np.arange(m, len(y)):
        f += [np.mean(y[s - m : s])]
    return np.array(f)


def get_loss_diff(y, f, m):
    loss1 = y[m:] ** 2
    loss2 = (y[m:] - f) ** 2
    return loss1 - loss2
```

Next, define the functions that compute the GW statistic and the one that computes the subsample t-test for given block size:


```python
def get_j_gw(loss_diff):
    return np.sum(loss_diff) / np.sqrt(np.sum(loss_diff ** 2))


def get_s_k(loss_diff, k):
    blocks = np.array_split(loss_diff, k)
    block_means = [np.mean(ld) for ld in blocks]
    grand_mean = np.mean(block_means)
    return (
        np.sqrt(k)
        * grand_mean
        / np.sqrt(1 / (k - 1) * np.sum((block_means - grand_mean) ** 2))
    )
```

Finally, write a wrapper to simulate values of the test statistics for a given parameter constellation:


```python
def simulate(T, m, n_sim, k, seed=None):
    np.random.seed(seed)
    df_sim = pd.DataFrame()
    p = get_p(m)
    for i in range(n_sim):
        y = np.random.binomial(1, p=p, size=int(T))
        f = rolling_forecast(y, m)
        loss_diff = get_loss_diff(y, f, m)
        j_gw = get_j_gw(loss_diff)
        s_k = get_s_k(loss_diff, k)
        df_sim = df_sim.append(
            pd.DataFrame({"j_gw": [j_gw], "s_k": [s_k], "m": m, "p": p, "T": T})
        )
    return df_sim
```

#### Run simulation

In the simulation, we use 10000 simulation per configuration. Configurations differ by window length and sample length. 


```python
df_config = pd.DataFrame(product([1, 3, 10, 20], [100, 500, 1000]), columns=["m", "T"])
n_sim = 10000
k = 20
```


```python
df_sim = pd.DataFrame()
```


```python
warnings.filterwarnings("ignore", category=RuntimeWarning)
for _, row in df_config.iterrows():
    df_sim_i = simulate(row["T"], row["m"], n_sim, k=k, seed=42)
    df_sim = pd.concat([df_sim, df_sim_i], ignore_index=True)
    # print(f"Finished simulation for m={row['m']}, T={row['T']}, p={get_p(row['m'])}")
```

### Results

#### Rejection frequencies for GW test

As expected, the GW has severe size distortions (i.e., rejection frequencies far from 0.05 at the critical value implied by the hypothesized limit distribution). Size distortions only appear when $p\neq0.5$ and become more severe as $p$ moves close to $0$. This is in line with ZT's finding that size distortions are related to skewness. Interestingly, the GW statistic suffers from undersizing here, i.e., too few rejections.


```python
df_sim.groupby(["m", "p", "T"])["j_gw"].agg(lambda x: np.mean(np.abs(x) > 1.96))
```




    m   p         T   
    1   0.500000  100     0.0540
                  500     0.0478
                  1000    0.0489
    3   0.250000  100     0.0104
                  500     0.0048
                  1000    0.0041
    10  0.090909  100     0.0471
                  500     0.0054
                  1000    0.0022
    20  0.047619  100     0.1704
                  500     0.0133
                  1000    0.0049
    Name: j_gw, dtype: float64



#### Rejection frequencies for subsample t test

The critical value for the subsample t test is the t distribution with $K-1$ degrees of freedom:


```python
cv_s_k = stats.t.ppf(q=0.975, df=k - 1)
cv_s_k
```




    2.093024054408263



Rejection frequencies under this critical value are not too far from the nominal level for sample sizes higher than 100, with slight oversizing. This is again in line with ZT's simulations results:


```python
df_sim.groupby(["m", "p", "T"])["s_k"].agg(lambda x: np.mean(np.abs(x) > cv_s_k))
```




    m   p         T   
    1   0.500000  100     0.0610
                  500     0.0472
                  1000    0.0498
    3   0.250000  100     0.0539
                  500     0.0551
                  1000    0.0527
    10  0.090909  100     0.0442
                  500     0.0624
                  1000    0.0630
    20  0.047619  100     0.1485
                  500     0.0591
                  1000    0.0611
    Name: s_k, dtype: float64



#### Distributions of test statistics

First, define a function to plot kernel density estimates of the test statistic next to a pdf of the hypothesized limit distribution.


```python
def plot_dists(
    df_sim,
    stat="GW",
    k=20,
    xlower=-3,
    xupper=3,
    palette=sns.color_palette("husl"),
    figsize=(16, 12),
):
    x = np.linspace(xlower, xupper, 100)
    Ts = df_sim["T"].unique()
    ps = df_sim["p"].unique()
    fig, axs = plt.subplots(len(Ts), len(ps), figsize=figsize)
    for i, T in enumerate(Ts):
        for j, p in enumerate(ps):
            df_sim_i = df_sim.loc[(df_sim["T"] == T) & (df_sim["p"] == p)]
            sns.kdeplot(
                df_sim_i["j_gw"] if stat == "GW" else df_sim_i["s_k"],
                legend=False,
                ax=axs[i, j],
                color=palette[0],
            )
            axs[i, j].set_xlim(-3, 3)
            axs[i, j].set_title(
                f"m={df_sim_i['m'].unique()[0]},p={np.round(p, 4)}, T={T}"
            )
            if stat == "GW":
                y = stats.norm.pdf(x)
            else:
                y = stats.t.pdf(x, df=k - 1)
            axs[i, j].plot(x, y, color=palette[1], ls=":", lw=1)
    fig.tight_layout()
```

#### GW statistic

For $p=0.05$, the kernel density estimate of the distribution of the test statistic is very close to the theoretical one. As $p$ decreases, the two start to differ substantially.


```python
plot_dists(df_sim, stat="GW")
```


    
![png](conditional_accuracy_files/conditional_accuracy_28_0.png)
    


#### Subsample t-statistic

This one also differs more from the theoretical on as $p$ decreases, but it is still approximated much better by the t distribution than the distribution of the GW statistic is approximated by the normal.


```python
plot_dists(df_sim, stat="subsample")
```


    
![png](conditional_accuracy_files/conditional_accuracy_30_0.png)
    


### References 

Diebold, Francis X., and Robert S. Mariano. "Comparing predictive accuracy." _Journal of Business & Economic Statistics_ 13, no. 3 (1995): 253–263.

Giacomini, Raffaella, and Halbert White. "Tests of conditional predictive ability." _Econometrica_ 74, no. 6 (2006): 1545-1578.

Ibragimov, Rustam, and Ulrich K. Müller. "t-Statistic based correlation and heterogeneity robust inference." _Journal of Business & Economic Statistics_ 28, no. 4 (2010): 453-468.



```python

```
