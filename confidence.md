# Three Solutions to Calculate Confidence Intervals
Let $X_1,…,X_n$ denote a random sample of independent observations from a population with mean $\mu$ and variance $\sigma^2$. Let $\theta$ be the parameter of interest. 
According to Central Limit Theorem and the Delta Method (if necessary), we can find the asymptotic distribution of an estimator of $\theta$, 
denoted by $\hat{\theta}$. Note that $\hat{\theta}$ needs to be a function of sample mean $\overline{X}$.

For example, 

1. $X \sim N(\mu,\sigma^2)$, $\hat{\mu} = \overline{X}$, and  $\sqrt{n}(\overline{X}-\mu) \rightarrow N(0,\sigma^2)$
 
2. $X \sim Ber(p)$, $\hat{p} = \overline{X}$, and  $\sqrt{n}(\overline{X}-p) \rightarrow N(0,p(1-p))$
  
3. $X \sim Exp(\lambda)$, $\hat{\lambda} = \dfrac{1}{\overline{X}}$, and  $\sqrt{n}(\dfrac{1}{\overline{X}}-\lambda) \rightarrow N(0,\lambda^2)$


A 95% ($\alpha=0.05$) confidence interval of $\theta$ can be defined by $\hat{\theta}\pm x$, where $x$ is a positive value such that
$$P(|\hat{\theta}-\theta|>x)<\alpha$$

Equivalently,
$$P\left(|\hat{\theta}-\theta|>x \right)=2P\left[(\hat{\theta}-\theta)>x) \right]=2\left(1-P[(\hat{\theta}-\theta) \leq x]\right)$$
$$=2\left[1-P\left(\dfrac{\sqrt{n}(\overline{\theta}-\theta)}{\sigma}\leq\dfrac{\sqrt{n}x}{\sigma}\right)\right] 
\textcolor{red}{=} 2 \left[1- \Phi \left(\dfrac{\sqrt{n}x}{\sigma}\right)\right] = \alpha$$
where $\Phi()$ is the CDF of standard Normal distribution.

The red equation is according to the Central Limit Theorem.
Since $$2 \left[1- \Phi \left(\dfrac{\sqrt{n}x}{\sigma}\right)\right] = \alpha$$
we have 
$$\dfrac{\sqrt{n}x}{\sigma} =\Phi^{-1}(1-\dfrac{\alpha}{2})=q_{1-\alpha/2}$$
Therefore
$$x=q_{1-\alpha/2}\dfrac{\sigma}{\sqrt{n}}$$
So the general form of a confidence interval of $\theta$ is (for $\alpha=0.05$)
$$\hat{\theta}\pm x = \hat{\theta} \pm q_{1-\alpha/2}\dfrac{\sigma}{\sqrt{n}} = \hat{\theta} \pm 1.96\dfrac{\sigma}{\sqrt{n}}$$

The problem with this form of confidence interval is, $\sigma$, which is usually a function of the true parameter of the underlying distribution, is unknown. 

## So we can use three solutions to get a confidence interval based on known information.
1. **Conservative Bound**: if $\sigma$ is bounded at $\sigma_0$, plug-in $\sigma_0$ to form the $CI_{cons}$
  $$CI_{cons} = \hat{\theta} \pm q_{1-\alpha/2}\dfrac{\sigma_0}{\sqrt{n}}$$
2. Solve: If $\sigma$ is a function of $\theta$, i.e., $\sigma=f(\theta)$, solve the inequality
  $$\hat{\theta} - q_{1-\alpha/2}\dfrac{f(\theta)}{\sqrt{n}} \leq \theta  \leq \hat{\theta} + q_{1-\alpha/2}\dfrac{f(\theta)}{\sqrt{n}}$$
  The solution to this problem is $a \leq \theta \leq b$, then
  $$CI_{solve} = (a,b)$$
  Note that the this confidence interval $CI_{solve}$ is not necessarily centered on the estimate $\hat{\theta}$.
3. Plug in: If $\sigma$ is a function of $\theta$, i.e., $\sigma=f(\theta)$, plug in the estimate of $\theta$, (i.e., $\hat{\theta}$) to this function
   $$CI_{plug-in} = \hat{\theta} \pm q_{1-\alpha/2}\dfrac{f(\hat{\theta})}{\sqrt{n}}$$


















