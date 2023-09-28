# Three Solutions to Calculate Confidence Intervals
### ipynb file can be found [here](https://github.com/houzhj/Statistics/blob/main/ipynb/01_confidence_intervals.ipynb)

Let $X_1,…,X_n$ denote a random sample of independent observations from a population with mean $\mu$ and variance $\sigma^2$. Let $\theta$ be the parameter of interest. 
According to Central Limit Theorem and the Delta Method (discussed [here](https://github.com/houzhj/Statistics/blob/main/01_Delta_Method.md)) (if necessary), we can find the asymptotic distribution of an estimator of $\theta$, 
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
2. **Solve**: If $\sigma$ is a function of $\theta$, i.e., $\sigma=f(\theta)$, solve the inequality
  $$\hat{\theta} - q_{1-\alpha/2}\dfrac{f(\theta)}{\sqrt{n}} \leq \theta  \leq \hat{\theta} + q_{1-\alpha/2}\dfrac{f(\theta)}{\sqrt{n}}$$
  The solution to this problem is $a \leq \theta \leq b$, then
  $$CI_{solve} = (a,b)$$
  Note that the this confidence interval $CI_{solve}$ is not necessarily centered on the estimate $\hat{\theta}$.
3. **Plug in**: If $\sigma$ is a function of $\theta$, i.e., $\sigma=f(\theta)$, plug in the estimate of $\theta$, (i.e., $\hat{\theta}$) to this function
   $$CI_{plug-in} = \hat{\theta} \pm q_{1-\alpha/2}\dfrac{f(\hat{\theta})}{\sqrt{n}}$$

### Now consider a few examples from different distributions.
## Example A - Bernoulli Distribution
$$X \sim Ber(p), \hat{p} = \overline{X}$$

By central limit theorem, $\sqrt{n}(\overline{X}-p) \rightarrow N(0,p(1-p))$. 

The confidence interval can be written as 
$$\overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$
1. **Conservative Bound**:
   
   Since $\sqrt{p(1-p)}\leq\sqrt{0.5(1-0.5)}=0.5$ when $p \in (0,1)$, we have
   $$CI_{cons} = \overline{X} \pm q_{1-\alpha/2}\dfrac{0.5}{\sqrt{n}}$$
   
2. **Solve**

   According to the following derivation
   $$\overline{X} - q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}} \leq  p \leq \overline{X} + q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$
   $$\downarrow$$
   $$- q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}} \leq  p-\overline{X} \leq  + q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$
   $$\downarrow$$
   $$(p-\overline{X})^2 \leq  \left( q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}\right)^2$$
   $$\downarrow$$
   $$Ap^2+Bp+c \leq 0$$
   where
   $$A=1+\dfrac{(q_{1-\alpha/2})^2}{n}, B=-2\overline{X}-\dfrac{(q_{1-\alpha/2})^2}{n}, C=(\overline{X})^2$$
   $$\downarrow$$
   $$CI_{solve} = \left( \dfrac{-B \pm \sqrt{B^2-4AC}}{2A} \right)$$

3. **Plug-in**
   Given that $\hat{p}=\overline{X}$
   $$CI_{plug-in} = \overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{n}} =\overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{\overline{X}(1-\overline{X})}}{\sqrt{n}}$$


Now, calculate the three types of confidence intervals using simulated data. Assume the true value of p is 0.6 (this information is unknown in reality, so it will be used in the derivation of confidence interval). Consider a level 90% for Confidence Intervals (i.e. $\alpha$=10%)

```python
### True value(s) of distribution parameter(s)
true_p     = 0.6

### True values of mean and variance of the distribution, derived based on the true value of parameter
true_mean  = true_p
true_var   = true_p*(1-true_p)
true_sigma = np.sqrt(true_var)

### Pre-specified significant level
alpha      = 0.1

### quantile value, will be used in the calculation of confidence intervals
q          = norm.ppf(1-alpha/2)

population = np.random.binomial(n=1, p=true_mean, size=100000)

n_experiment = 1000
sample_size  = 100

### A dataframe used to record whether the real values are within the derived confidence intervals.
true_value_in_ci = pd.DataFrame({'conservative':[np.nan]*n_experiment,
                                 'solve':[np.nan]*n_experiment,
                                 'plugin':[np.nan]*n_experiment})

### A dataframe used to record the boundaries of the confidence intervals.
ci_results       = pd.DataFrame({'conservative_l':[np.nan]*n_experiment,
                                 'conservative_r':[np.nan]*n_experiment,
                                 'solve_l':[np.nan]*n_experiment,
                                 'solve_r':[np.nan]*n_experiment,
                                 'plugin_l':[np.nan]*n_experiment,
                                 'plugin_r':[np.nan]*n_experiment,
                                 'conservative_range':[np.nan]*n_experiment,
                                 'solve_range':[np.nan]*n_experiment,
                                 'plugin_range':[np.nan]*n_experiment,
                                })
```
In the loop below, we conducted 1000 experiments. In each experiment, we employed the same method to construct a Confidence Interval:
- Step 1. A random sample was drawn from the population, with the pre-defined sample size.
- Step 2. Based on the aforementioned formulas, Confidence Intervals were computed using three methods (if applicable).

As a result, we obtained 1000 Confidence Intervals. These Confidence Intervals do not depend on the true parameter but depend on the data within the samples. Their boundaries and widths varied a bit across different experiments (with the exception of the conservative Confidence Interval, which solely depends on the sample size in this Bernoulli case).
According to the definition to a Confidence Intervals of level 90%, we can anticipate that in these 1000 experiments, the true parameter will fall within the Confidence Interval in at least 90% of the cases. We verified this is true for all the three types of the Confidence Intervals. We also compared the widths of different Confidence Intervals.

```python
for i in range(n_experiment):
    sample       = np.random.choice(population,size=sample_size,replace=True)
    sample_mean  = np.mean(sample)
    ##### Conservative CI
    ci_conservative_l = sample_mean-q*0.5/np.sqrt(sample_size)
    ci_conservative_r = sample_mean+q*0.5/np.sqrt(sample_size)
    ci_results.loc[i,'conservative_l'] = ci_conservative_l
    ci_results.loc[i,'conservative_r'] = ci_conservative_r
    #print(ci_conservative_l,ci_conservative_r)
    ##### Solve CI
    A            = 1+q**2/sample_size
    B            = -2*sample_mean-q**2/sample_size
    C            = sample_mean**2
    ci_solve_l    = (-B-np.sqrt(B**2-4*A*C))/(2*A)
    ci_solve_r    = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    ci_results.loc[i,'solve_l'] = ci_solve_l
    ci_results.loc[i,'solve_r'] = ci_solve_r
    #print(ci_solve_l,ci_solve_r)
    
    ##### Solve Plug-in 
    ci_plugin_l  = sample_mean-q*np.sqrt(sample_mean*(1-sample_mean))/np.sqrt(sample_size)
    ci_plugin_r  = sample_mean+q*np.sqrt(sample_mean*(1-sample_mean))/np.sqrt(sample_size)
    ci_results.loc[i,'plugin_l'] = ci_plugin_l
    ci_results.loc[i,'plugin_r'] = ci_plugin_r
    #print(ci_plugin_l,ci_plugin_r)
              
    true_value_in_ci.loc[i,'conservative'] = int((ci_conservative_l<=true_p) &(ci_conservative_r>=true_p))
    true_value_in_ci.loc[i,'solve'] = int((ci_solve_l<=true_p) &(ci_solve_r>=true_p))
    true_value_in_ci.loc[i,'plugin'] = int((ci_plugin_l<=true_p) &(ci_plugin_r>=true_p))           
```
Percentage of the experiments in which the true parameter falls within the confidence intervals
```python
temp = pd.DataFrame(columns=['method','% of true parameter falls within CI'])
temp.iloc[:,0] = ['conservative','solve','plugin']
temp.iloc[:,1] = list(true_value_in_ci.mean())
temp
```
<img width="353" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/d5fe3f0f-d06f-448b-81b5-f9f55ddeea87">

Comparing the width of the confidence intervals derived using different methods
```python
ci_results['conservative_range'] = ci_results['conservative_r'] -ci_results['conservative_l'] 
ci_results['solve_range']        = ci_results['solve_r'] -ci_results['solve_l'] 
ci_results['plugin_range']       = ci_results['plugin_r'] -ci_results['plugin_l'] 
ci_results['widest']             = ci_results[['conservative_range', 
                                               'solve_range', 
                                               'plugin_range']].apply(lambda x: x.idxmax(), axis=1)
ci_results['narrowest']          = ci_results[['conservative_range', 
                                               'solve_range', 
                                               'plugin_range']].apply(lambda x: x.idxmin(), axis=1)
ci_results.head().round(4)
```


In all experiments, "plugin" CIs are the narrowest.

<img width="902" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/aa27f22b-d30d-4ebd-b7bb-02a6d1ce365e">


## Example B - Exponential Distribution
$$X \sim Exp(\lambda), \hat{\lambda} = \dfrac{1}{\overline{X}}$$

By central limit theorem and delta method, $\sqrt{n}(\dfrac{1}{\overline{X}}-\lambda) \rightarrow N(0,\lambda^2)$. 

The confidence interval can be written as 
$$\dfrac{1}{\overline{X}} \pm q_{1-\alpha/2} \dfrac{\sqrt{\lambda^2)}}{\sqrt{n}} = \dfrac{1}{\overline{X}} \pm q_{1-\alpha/2} \dfrac{\lambda}{\sqrt{n}}$$

1. **Conservative Bound**:
   
   $\lambda>0$ is not bounded, so 
   $$CI_{cons} = (-\infty,\infty)$$
   
2. **Solve**

   According to the following derivation
   $$\hat{\lambda} - q_{1-\alpha/2} \dfrac{\lambda}{\sqrt{n}} \leq  \lambda \leq \hat{\lambda} + q_{1-\alpha/2} \dfrac{\lambda}{\sqrt{n}}$$
   $$\downarrow$$
   $$\lambda \geq \hat{\lambda} \left( 1+ \dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1},\lambda \leq \hat{\lambda} \left(1-\dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1}$$
   $$\downarrow$$
   $$CI_{solve} = \left (\hat{\lambda} \left( 1+ \dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1}, \hat{\lambda} \left(1-\dfrac{q_{1+\alpha/2}}{\sqrt{n}} \right)^{-1} \right)
   = \left (\dfrac{1}{\overline{X}} \left( 1+ \dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1}, \dfrac{1}{\overline{X}}\left(1-\dfrac{q_{1+\alpha/2}}{\sqrt{n}} \right)^{-1} \right)$$

3. **Plug-in**
   Given that $\hat{/lambda}=\dfrac{1}{\overline{X}}$
   $$CI_{plug-in} = \dfrac{1}{\overline{X}} \pm q_{1-\alpha/2} \dfrac{\hat{\lambda}}{\sqrt{n}}=\left(
   \dfrac{1}{\overline{X}} \left(1-\dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right) , \dfrac{1}{\overline{X}} \left(1+\dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)\right)$$

Then calculate the three types of confidence intervals using simulated data. Assume the true value of $\lambda$ is 3. Consider a level 90% for Confidence Intervals (i.e. $\alpha$=10%). The codes are similar with those for Example A.

```python
image.png
### True value(s) of distribution parameter(s)
true_Lambda = 3

### True values of mean and variance of the distribution, derived based on the true value of parameter
true_mean   = 1/true_Lambda
true_var    = 1/(true_Lambda**2)
true_sigma  = np.sqrt(true_var)

### Pre-specified significant level
alpha       = 0.1

### quantile value, will be used in the calculation of confidence intervals
q           = norm.ppf(1-alpha/2)

##### In np.random.exponential, to create a sample of Exp(Lambda), using scale = 1/Lambda
population = np.random.exponential(scale = 1/true_Lambda, size=100000)

n_experiment = 1000
sampel_size  = 100

true_value_in_ci = pd.DataFrame({'conservative':[np.nan]*n_experiment,
                                 'solve':[np.nan]*n_experiment,
                                 'plugin':[np.nan]*n_experiment})
ci_results       = pd.DataFrame({'conservative_l':[np.nan]*n_experiment,
                                 'conservative_r':[np.nan]*n_experiment,
                                 'solve_l':[np.nan]*n_experiment,
                                 'solve_r':[np.nan]*n_experiment,
                                 'plugin_l':[np.nan]*n_experiment,
                                 'plugin_r':[np.nan]*n_experiment,
                                 'conservative_range':[np.nan]*n_experiment,
                                 'solve_range':[np.nan]*n_experiment,
                                 'plugin_range':[np.nan]*n_experiment,
                                })

for i in range(n_experiment):
    sample       = np.random.choice(population,size=sample_size,replace=True)
    sample_mean  = np.mean(sample)
    ##### Conservative CI
    ci_conservative_l = -99999
    ci_conservative_r = 99999
    ci_results.loc[i,'conservative_l'] = ci_conservative_l
    ci_results.loc[i,'conservative_r'] = ci_conservative_r
    #print(ci_conservative_l,ci_conservative_r)
    ##### Solve CI
    ci_solve_l    = (1/sample_mean)/(1+q/np.sqrt(sample_size))
    ci_solve_r    = (1/sample_mean)/(1-q/np.sqrt(sample_size))
    ci_results.loc[i,'solve_l'] = ci_solve_l
    ci_results.loc[i,'solve_r'] = ci_solve_r
    #print(ci_solve_l,ci_solve_r)
    
    ##### Solve Plug-in 
    ci_plugin_l  = (1/sample_mean)*(1-q/np.sqrt(sample_size))
    ci_plugin_r  = (1/sample_mean)*(1+q/np.sqrt(sample_size))
    ci_results.loc[i,'plugin_l'] = ci_plugin_l
    ci_results.loc[i,'plugin_r'] = ci_plugin_r
    #print(ci_plugin_l,ci_plugin_r)
              
    true_value_in_ci.loc[i,'conservative'] = int((ci_conservative_l<=true_Lambda) &(ci_conservative_r>=true_Lambda))
    true_value_in_ci.loc[i,'solve'] = int((ci_solve_l<=true_Lambda) &(ci_solve_r>=true_Lambda))
    true_value_in_ci.loc[i,'plugin'] = int((ci_plugin_l<=true_Lambda) &(ci_plugin_r>=true_Lambda))     
```

Percentage of the experiments in which the true parameter falls within the confidence intervals

<img width="353" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/0b0383bc-2ce6-4afa-9875-570cfed3be6b">


```python
temp = pd.DataFrame(columns=['method','% of true parameter falls within CI'])
temp.iloc[:,0] = ['conservative','solve','plugin']
temp.iloc[:,1] = list(true_value_in_ci.mean())
temp
```

Comparing the width of the confidence intervals derived using different methods
```python
ci_results['conservative_range'] = ci_results['conservative_r'] -ci_results['conservative_l'] 
ci_results['solve_range'] = ci_results['solve_r'] -ci_results['solve_l'] 
ci_results['plugin_range'] = ci_results['plugin_r'] -ci_results['plugin_l'] 
ci_results['widest']    = ci_results[['conservative_range',
                                      'solve_range',
                                      'plugin_range']].apply(lambda x: x.idxmax(), axis=1)
ci_results['narrowest'] = ci_results[['conservative_range',
                                      'solve_range',
                                      'plugin_range']].apply(lambda x: x.idxmin(), axis=1)
ci_results.head().round(4)
```

In all experiments, "plugin" CIs are the narrowest.

<img width="907" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/6c9b7028-e6fb-4008-a1f5-cd419e236e62">




## Example C - Gamma Distribution
$$X \sim Gamma(\alpha,1/\alpha), \hat{\alpha} = \sqrt{\overline{X}}$$
Note that there is a simplified setting that $\beta = 1/\alpha$, which is not necessarioy the case. So there is only one unknown parameter.

By central limit theorem and delta method, $\sqrt{n}(\sqrt{\overline{X}}-\alpha) \rightarrow N(0,\alpha/4)$. 

The confidence interval can be written as (let $q=q_{1-\alpha/2}$ to avoid two duplicated $\alpha$)
$$\sqrt{\overline{X}} \pm q \dfrac{\sqrt{\alpha/4}}{\sqrt{n}} = \sqrt{\overline{X}} \pm \dfrac{q\sqrt{\alpha}}{2\sqrt{n}}$$

1. **Conservative Bound**:
   
   $\alpha>0$ is not bounded, so 
   $$CI_{cons} = (-\infty,\infty)$$
   
2. **Solve**

   According to the following derivation
   $$\sqrt{\overline{X}} - \dfrac{q\sqrt{\alpha}}{2\sqrt{n}} \leq \alpha \leq \sqrt{\overline{X}} + \dfrac{q\sqrt{\alpha}}{2\sqrt{n}}$$
   $$\downarrow$$
   $$\alpha - \sqrt{\overline{X}} \leq \dfrac{q\sqrt{\alpha}}{2\sqrt{n}}$$
   $$\downarrow$$
   $$\left( \alpha - \sqrt{\overline{X}} \right)^2  \leq \left( \dfrac{q\sqrt{\alpha}}{2\sqrt{n}} \right) ^2 = \dfrac{q^2\alpha}{4n}$$
   $$\downarrow$$
   $$Ap^2+Bp+c \leq 0$$
   where
   $$A=1, B=-2\sqrt{\overline{X}}-\dfrac{q^2}{4n}, C=\overline{X}$$
   $$\downarrow$$
   $$CI_{solve} = \left( \dfrac{-B \pm \sqrt{B^2-4AC}}{2A} \right)$$

   
3. **Plug-in**
   
   Given that $\hat{\lambda}=\dfrac{1}{\overline{X}}$
   $$CI_{plug-in} = \sqrt{\overline{X}} \pm \dfrac{q \sqrt{\hat{\alpha}}}{2\sqrt{n}} = \sqrt{\overline{X}} \pm \dfrac{q\sqrt{\sqrt{\overline{X}}}}{2\sqrt{n}}$$

Then calculate the three types of confidence intervals using simulated data. Assume the true value of $\alpha$ is 3. Consider a level 90% for Confidence Intervals (i.e. $\alpha$=10%). The codes are similar with those for Example A and Example B.

```python
### True value(s) of distribution parameter(s)
true_Alpha  = 3
## below is a simplified setting (keep the number of parameters to be 1), not necessarily the case
true_Beta   = 1/true_Alpha 

### True values of mean and variance of the distribution, derived based on the true value of parameter
true_mean   = true_Alpha**2
true_var    = true_Alpha**3
true_sigma  = np.sqrt(true_var)

### Pre-specified significant level
alpha       = 0.1

### quantile value, will be used in the calculation of confidence intervals
q           = norm.ppf(1-alpha/2)

##### In scipy.stats.gamma, to create a sample of Gamma(Alpha,Beta), using a=Alpha, scale = 1/Beta
population = gamma_samples = gamma.rvs(a=true_Alpha, scale=true_Alpha, size=100000)

n_experiment = 1000
sampel_size  = 100

true_value_in_ci = pd.DataFrame({'conservative':[np.nan]*n_experiment,
                                 'solve':[np.nan]*n_experiment,
                                 'plugin':[np.nan]*n_experiment})
ci_results       = pd.DataFrame({'conservative_l':[np.nan]*n_experiment,
                                 'conservative_r':[np.nan]*n_experiment,
                                 'solve_l':[np.nan]*n_experiment,
                                 'solve_r':[np.nan]*n_experiment,
                                 'plugin_l':[np.nan]*n_experiment,
                                 'plugin_r':[np.nan]*n_experiment,
                                 'conservative_range':[np.nan]*n_experiment,
                                 'solve_range':[np.nan]*n_experiment,
                                 'plugin_range':[np.nan]*n_experiment,
                                })

for i in range(n_experiment):
    sample       = np.random.choice(population,size=sample_size,replace=True)
    sample_mean  = np.mean(sample)
    ##### Conservative CI
    ci_conservative_l = -99999
    ci_conservative_r = 99999
    ci_results.loc[i,'conservative_l'] = ci_conservative_l
    ci_results.loc[i,'conservative_r'] = ci_conservative_r
    #print(ci_conservative_l,ci_conservative_r)
    ##### Solve CI
    A            = 1
    B            = -2*np.sqrt(sample_mean)-q**2/(4*sample_size)
    C            = sample_mean
    ci_solve_l    = (-B-np.sqrt(B**2-4*A*C))/(2*A)
    ci_solve_r    = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    ci_results.loc[i,'solve_l'] = ci_solve_l
    ci_results.loc[i,'solve_r'] = ci_solve_r
    #print(ci_solve_l,ci_solve_r)
    
    ##### Solve Plug-in 
    ci_plugin_l  = np.sqrt(sample_mean)-q*np.sqrt(np.sqrt(sample_mean))/(2*np.sqrt(sample_size))
    ci_plugin_r  = np.sqrt(sample_mean)+q*np.sqrt(np.sqrt(sample_mean))/(2*np.sqrt(sample_size))
    ci_results.loc[i,'plugin_l'] = ci_plugin_l
    ci_results.loc[i,'plugin_r'] = ci_plugin_r
    #print(ci_plugin_l,ci_plugin_r)
              
    true_value_in_ci.loc[i,'conservative'] = int((ci_conservative_l<=true_Lambda) &(ci_conservative_r>=true_Lambda))
    true_value_in_ci.loc[i,'solve'] = int((ci_solve_l<=true_Lambda) &(ci_solve_r>=true_Lambda))
    true_value_in_ci.loc[i,'plugin'] = int((ci_plugin_l<=true_Lambda) &(ci_plugin_r>=true_Lambda))      

```

Percentage of the experiments in which the true parameter falls within the confidence intervals
```python
temp = pd.DataFrame(columns=['method','% of true parameter falls within CI'])
temp.iloc[:,0] = ['conservative','solve','plugin']
temp.iloc[:,1] = list(true_value_in_ci.mean())
temp
```
<img width="346" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/145ab8c5-6cea-40fa-9845-2bb53e1a9cd5">

Comparing the width of the confidence intervals derived using different methods
```python
ci_results['conservative_range'] = ci_results['conservative_r'] -ci_results['conservative_l'] 
ci_results['solve_range'] = ci_results['solve_r'] -ci_results['solve_l'] 
ci_results['plugin_range'] = ci_results['plugin_r'] -ci_results['plugin_l'] 
ci_results['widest']    = ci_results[['conservative_range',
                                      'solve_range',
                                      'plugin_range']].apply(lambda x: x.idxmax(), axis=1)
ci_results['narrowest'] = ci_results[['conservative_range',
                                      'solve_range',
                                      'plugin_range']].apply(lambda x: x.idxmin(), axis=1)
ci_results.head().round(4)
```

In all experiments, "plugin" CIs are the narrowest.

<img width="891" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/c8fcfeea-acc0-47df-9e1e-b126d884eded">





