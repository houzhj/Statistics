
# Delta Method
### ipynb file can be found [here]()

# 1. The Wald Test
The Wald statistic (for a single parameter $\theta$) takes the following form:
$$W = \dfrac{(\hat{\theta}-\theta_0)^2}{var(\hat{\theta})}$$
Under the null $H_0: \theta = \theta_0$ the test statistic $W \sim \chi^2(1)$

An alternative expression of the $W$ statistic is the square root of the one above, i.e.,
$$W = \dfrac{(\hat{\theta}-\theta_0)}{\sqrt{var(\hat{\theta})}}=\dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})}$$

Under the null $H_0: \theta = \theta_0$ the test statistic $W \sim N(0,1)$

Now focus on this second expression.

# 2. Likelihood Test
The likelihood ratio test basically looks at how much more likely is the alternative hypothesis when compared to the null hypothesis. To do this we look at the likelihood ratio (LR), consider the ratio

$$LR =\dfrac{sup_{\theta \in \Theta} L(X_1,...,X_n|\Theta)}{sup_{\theta \in \Theta_0} L(X_1,...,X_n|\Theta_0)} > c > 1$$
where 
- the $\sup$ notation refers to the [supremum](https://en.wikipedia.org/wiki/Infimum_and_supremum)
- $\Theta$ is the entire parameter space
- $\Theta_0$ is the parameter space specified in $H_0$, which is a subset of $\Theta$
- $c$ is a constant determined by the significance level of our test and that constant is greater than 1 (becasue $\Theta_0 \in \Theta$).

If this ratio suggests that the alternative hypothesis (numerator) is much more likely than the null hypothesis (denominator), then we want to reject the null hypothesis. This can be rewritten as the following test statistic (the reason for taking log and multiplying by 2 is because we will use the Wilks' theorem)
$$2 \times ln \left[\dfrac{sup_{\theta \in \Theta} L(X_1,...,X_n|\Theta)}{sup_{\theta \in \Theta_0} L(X_1,...,X_n|\Theta_0)}\right]$$

which gives rises to the test statistic of a LR Test (expressed as a difference between the log-likelihoods
)
$$T_n = 2\times [\mathcal{l}(\hat{\theta})-\mathcal{l}(\theta)]$$

According to [Wilks’ theorem](https://en.wikipedia.org/wiki/Wilks%27_theorem), if $H_0$ is true, $T_n$ will be asymptotically chi-squared distributed with degrees of freedom equal to the difference in dimensionality of $\Theta$ and $\Theta_0$


# 3. Examples
Next, we will perform and compare the Wald tests and the Likelihood Ratio (LR) tests through simulations. 
- We will consider several examples of hypothesis tests regarding parameters from different distributions.
  - Bernoulli distribution
  - Binomial distribution
  - Poisson distribution
  - Uniform distribution
- In each example, we will use a true null hypothesis as well as several false null hypotheses, to conduct experiments to calculate the probabilities of **Type 1 Error** and **Type 2 Error** for both testing methods.
- Additionally, we will investigate the impact of sample size on the tests.
- The significant level for all the tests are 5%.
- All the tests are two-sided, i.e., $H_0: \theta=$ certain value. 
- **Remark: Note that in this analysis, when we perform data generation and calculate type 1 and type 2 errors, we know the true values of the parameters. This differs from conducting hypothesis tests in real-world scenarios.**


## Example 1 - Bernoulli(0.4)
The codes below calculate the Wald test statistic using simulated data
- This is a distribution specific function. For example, this one is for Bernoulli distribution, meaning the test statistic and data generation are specific for Bernoulli distribution. 
- One can specifies the sample size
- One can specifies the true parameter (p_true) and the parameter in the null hypothesis (h0). If p_true = h0, the null is true; otherwise, the null is false.
- The notes abovo apply to many functions below as well. 

```python
def wald_ber_two_side(sample_size,h0,p_true,alpha=0.05):
    critical_value = norm.ppf(1-alpha/2)
    result = pd.DataFrame(columns=['w','sample_size','h_0','reject'])
    n_experiment = 1000
    w            = [np.nan]*n_experiment
    reject       = [np.nan]*n_experiment
    for i in range(n_experiment):
        sample      = np.random.binomial(n=1,p=p_true,size=sample_size)
        sample_mean = np.mean(sample)
        phat        = sample_mean
        w[i]        = abs((phat-h0)/np.sqrt(sample_mean*(1-sample_mean)/sample_size))
        reject[i] = int(w[i]>=critical_value)
    result['w']           = w
    result['sample_size'] = sample_size
    result['h_0']         = h0
    result['reject']      = reject
    return(result)
```

The codes below calculate the LR test statistic using simulated data. 
```python
def lr_ber_two_side(sample_size,h0,p_true,alpha=0.05):
    critical_value = chi2.ppf(1-alpha, 1)
    result = pd.DataFrame(columns=['lr','sample_size','h_0','reject'])
    n_experiment = 1000
    lr           = [np.nan]*n_experiment
    reject       = [np.nan]*n_experiment
    for i in range(n_experiment):
        sample      = np.random.binomial(n=1,p=p_true,size=sample_size)
        sample_mean = np.mean(sample)
        phat        = sample_mean
        sample_sum  = np.sum(sample) 
        lr[i] =2*(np.log(phat/h0)*sample_sum+np.log((1-phat)/(1-h0))*(sample_size-sample_sum))
        reject[i] = int(lr[i]>=critical_value)
    result['lr']          = lr
    result['sample_size'] = sample_size
    result['h_0']         = h0
    result['reject']      = reject
    return(result)
```

The codes below conduct and compare the Wald tests and the LR tests, with given sample size and given hypotheses. 
```python
def wald_lr_compare(sample_size_list,h0,p_true):
    compare_result = pd.DataFrame(columns = ['sample_size','true_p','h0',
                                             'rejection_rate_wald','rejection_rate_lr',
                                            ])
    for ss in range(len(sample_size_list)):
        ss_now = sample_size_list[ss]
        
        compare_result.loc[ss,'sample_size']    = ss_now
        compare_result.loc[ss,'true_p']         = p_true
        compare_result.loc[ss,'h0']             = h0
        
        result_wald = wald_ber_two_side(sample_size=ss_now, h0=h0, p_true=p_true)
        reject_rate_wald = result_wald['reject'].mean()
        compare_result.loc[ss,'rejection_rate_wald'] = reject_rate_wald
        result_lr   = lr_ber_two_side(sample_size=ss_now, h0=h0, p_true=p_true)
        reject_rate_lr   = result_lr['reject'].mean()
        compare_result.loc[ss,'rejection_rate_lr'] = reject_rate_lr

        if h0==p_true:
            error_rate_wald = reject_rate_wald
            error_rate_lr   = reject_rate_lr
            compare_result.loc[ss,'type_1_error_wald'] = error_rate_wald
            compare_result.loc[ss,'type_1_error_lr']   = error_rate_lr
        else: 
            error_rate_wald = 1 - reject_rate_wald
            error_rate_lr   = 1 - reject_rate_lr
            compare_result.loc[ss,'type_2_error_wald'] = error_rate_wald
            compare_result.loc[ss,'type_2_error_lr']   = error_rate_lr
    return(compare_result)
```
### (1) False Null Hypothesis
Let's first consider the following sample sizes: [20,50,100,200,400,1000], and the following null hypotheses (which are false)
- True value: $p = 0.4$
- $H_0: p=0.41$, $H_1: p \ne 0.41$
- $H_0: p=0.43$, $H_1: p \ne 0.43$
- $H_0: p=0.45$, $H_1: p \ne 0.45$
- $H_0: p=0.5$, $H_1: p \ne 0.5$
- $H_0: p=0.6$, $H_1: p \ne 0.6$

We conduct a comparison analysis for each combination of sample size and null hypothesis. 

```python
sample_size_list = [20,50,100,200,400,1000]
h0_list          = [0.41,0.43,0.45,0.5,0.6]
p_true           = 0.4
for h0 in h0_list:
    compare_now = wald_lr_compare(sample_size_list,h0,p_true)
    if h0==h0_list[0]:
        compare_final = compare_now
    else:
        compare_final = pd.concat([compare_final,compare_now],axis=0)
```
The results are stored in a dataframe, but let's show them by plots. 
<img width="976" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/1ab19b0a-564d-40fa-bce8-0498d709160c">

The results suggest:
- The Type 2 Error rates of the two tests are close, with different sample sizes.
- As the sample size increases, the Type 2 Error (failing to reject a false null hypothesis) decreases.
- As the values in the false null hypothesis get closer to the true values, a larger sample size is required to reject the null hypothesis.

### (2) True Null Hypothesis
Consider the null hypothesis (which is true): $H_0 = 0.4, H_1 \ne 0.4$


```python
sample_size_list = [20,50,100,200,400,1000]
h0_list          = [0.4]
p_true      = 0.4
for h0 in h0_list:
    compare_now = wald_lr_compare(sample_size_list,h0,p_true)
    if h0==h0_list[0]:
        compare_final = compare_now
    else:
        compare_final = pd.concat([compare_final,compare_now],axis=0)

df_w = compare_final[['sample_size','h0','type_1_error_wald']].rename(columns={'type_1_error_wald':'type_1_error'})
df_w['Test Type'] = 'Wald'
df_lr = compare_final[['sample_size','h0','type_1_error_lr']].rename(columns={'type_1_error_lr':'type_1_error'})
df_lr['Test Type'] = 'LR'
combined = pd.concat([df_w,df_lr],axis=0) 
g = sns.FacetGrid(data=combined,hue='Test Type',col='h0')
g.map(sns.scatterplot,'sample_size','type_1_error',alpha=0.5)
g.add_legend()
plt.show()
```
<img width="260" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/616c0436-0295-44db-bc60-d80364d8cbad">

The plots show that the Type 1 Error rates of these two tests are generally close - around 5% to 10%.



## Example 2 - Binomial(5,0.4)

A Binomial distribution has two parameters, $K$ and $p$. Let take $K$ as given ($K$=5)

The codes for this and the the following examples can be found [here]().

### (1) False Null Hypothesis
<img width="979" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/0151c408-6fe3-4bf3-bc0c-e43e765a9c8a">

### (2) True Null Hypothesis
<img width="254" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/0779f20c-a9bf-4bd6-a106-ac49a788718d">


## Example 3 - Poisson(5)

### (1) False Null Hypothesis
<img width="986" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/aa65a13b-5c5f-4c8c-b77f-3ea2c512429b">


### (2) True Null Hypothesis
<img width="247" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/449b2a10-3808-469f-9a58-0b62427a6243">


## Example 4 - Uniform[0,b]
Consider a random variable $X \sim U[a,b]$ The maximum likelihood estimator of the upper bound parameter $b$ can be derived as below. 
$$L(a,b) = \prod_{i=1}^{n} \dfrac{1}{b-a} = (b-a)^{-n}$$
$$\downarrow$$
$$l(a,b) = -n \times ln(b-a)$$




