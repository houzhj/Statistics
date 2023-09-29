

# 3. Examples with Normal distribution 
Consider the Wald test for a $N(\mu,\sigma^2)$ distribution, with 
$$H_0: \mu = \mu_0, H_1:\mu \ne \mu_0$$

As discussed above, the Wald test statistic is
$$W = \dfrac{\hat{\mu}-\mu}{\sqrt{var(\hat{\mu})}} = \dfrac{\hat{\mu}-\mu_0}{se(\hat{\mu})}$$

The maximum likelihood estimate of $\mu$ is 
$$\hat{\mu} = \overline{X}$$
and the sample variance of $\hat{mu}$ is 
$$\widehat{Var}(\hat{\mu}) =\widehat{Var}(\overline{X}) = \widehat{Var}\left(\dfrac{1}{n} \sum_{i=1}^n X_i \right) =\dfrac{1}{n^2}\widehat{Var}\left(\sum_{i=1}^n X_i  \right)=\dfrac{1}{n^2}\times n \times \widehat{Var}(X_i) $$

$$ = \dfrac{1}{n}\widehat{Var}(\sigma^2) = \dfrac{1}{n} \times \left[\dfrac{1}{n-1}\sum_{i=1}^n \left(X_i-\overline{X}\right)^2\right] =\dfrac{1}{n} s_n^2$$

It can be proved (shown below) that $s_n^2$ is an unbiased estimator or $\sigma_2$, i.e., $E(s^2)=\sigma^2$

#### Proof:
$$E(s^2) = E\left[ \dfrac{1}{n-1}\sum_{i=1}^n \left(X_i-\overline{X}\right)^2   \right] = \dfrac{1}{n-1}E\left[ \sum_{i=1}^n X_i^2 +n(\overline{X})^2 - 2 \sum_{i=1}^n X_i \overline{X}\right]$$
$$ =\dfrac{1}{n-1}E\left[ \sum_{i=1}^n X_i^2 -n(\overline{X}^2) \right]= \dfrac{1}{n-1} \left[ nE(X_i^2)-nE(\overline{X}^2)\right]$$

$$ = \dfrac{n}{n-1} \left( \left[E(X_i)^2+Var(X_i)\right]-   \left[E(\overline{X})^2+Var(\overline{X})\right]   \right)  $$

$$ = \dfrac{n}{n-1} \left[  \left(\mu^2+\sigma^2\right)-\left(\mu^2 + \dfrac{1}{n}\sigma^2\right)  \right] = \dfrac{n}{n-1} \left( \dfrac{n-1}{n}\sigma^2\right)=\sigma^2$$

### QED

So the $W$ statistic in a given sample is calculated as 
$$W =  \dfrac{\hat{\mu}-\mu}{\sqrt{\widehat{var}(\hat{\mu})}} = \dfrac{\hat{\mu}-\mu}{\sqrt{s_n^2/n}} = \dfrac{\sqrt{n}(\overline{X}-\mu)}{\sqrt{s_n^2}}$$
where 
$$s_n^2 = \dfrac{1}{n-1} \sum_{i=1}^n \left( X_i-\overline{X}\right)^2$$

Under $H_0: \mu = \mu_0$, and $n$ is large enough 
$$W \rightarrow \dfrac{\sqrt{n}(\overline{X}-\mu)}{\sigma} \sim N(0,1)$
However, this does not hold when $n$ is small, see discussion below.

## 3.1 - Very Small Sample Size
Consider a case when there are only 2 observation. When $n=2$, $\overline{X} = \dfrac{1}{2}(X_1+X_2)$
$$s_n^2 = \dfrac{1}{2-1}\left[ \left( X_1 - \dfrac{1}{2}X_1- \dfrac{1}{2}X_2   \right)^2  + \left( X_2 - \dfrac{1}{2}X_1- \dfrac{1}{2}X_2   \right)^2  \right] = \dfrac{1}{2}(X_1-X_2)^2$$
$$\downarrow$$
$$W = \dfrac{\sqrt{n}(\overline{X}-\mu)}{\sqrt{s_n^2}} = \dfrac{\sqrt{2} \left( \dfrac{1}{2}X_1+\dfrac{1}{2}X_2 -\mu_0 \right)}{\sqrt{\dfrac{1}{2}(X_1-X_2)^2}}
= \dfrac{X_1+X_2-2\mu_0}{\sqrt{(X_1-X_2)^2}}$$












