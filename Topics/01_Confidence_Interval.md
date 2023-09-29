# Three Solutions to Calculate Confidence Intervals


Let $X_1,…,X_n$ denote an i.i.d random samples from a population with mean $\mu$ and variance $\sigma^2$.
$$E(X_i) = \mu$$
$$Var(X_i) = \sigma^2$$

Let $\theta$ be the parameter of interest. 
- Here we assume the estimation of $\theta$  can be expressed as a functuon of sample mean $\overline(X)$.
- According to Central Limit Theorem and the Delta Method (if necessary), we can find the asymptotic distribution of the estimator of $\theta$ (denoted by $\hat{\theta}$). 


For example, 

1. $X \sim N(\mu,\sigma^2)$, $\hat{\mu} = \overline{X}$, and  $\sqrt{n}(\overline{X}-\mu) \rightarrow N(0,\sigma^2)$
 
2. $X \sim Ber(p)$, $\hat{p} = \overline{X}$, and  $\sqrt{n}(\overline{X}-p) \rightarrow N(0,p(1-p))$
  
3. $X \sim Exp(\lambda)$, $\hat{\lambda} = \dfrac{1}{\overline{X}}$, and  $\sqrt{n}(\dfrac{1}{\overline{X}}-\lambda) \rightarrow N(0,\lambda^2)$

### The contents of this note
#### 1. Derivation of a general form of confidence interval for the parameter
#### 2. Three methods to find a confidence interval based on known information from the sample.
#### 3. Examples
