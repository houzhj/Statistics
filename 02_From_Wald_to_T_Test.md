# From Wald Test to Student's t test (T test)
### The contents of this note
- **Wald Test**
- **Examples with Bernoulli distribution**
  - **Wald test with very Small Sample Size**
  - **Simulation Analysis for Wald tests in Bernoulli distribution**
    - Experiment A. Very small sample size
    - Experiment B. Changing null hypothesis with fixed sample size(n=50)
    - Experiment C. Changing the sample size


$$$$

$$$$


## Part 1 - Wald Test
Wald test is a maximum likelihood estimate based test. It is based on the asymptotic normal approximation to the maximum likelihood estimator.
Two equivalent expression of the Wald test statistics are shown below
Consider the parameter $\theta$. The maximum likelihood of $\theta$ is $\hat{\theta}$
The hypotheses are given below (the hypothesis can be one-sided too)
$$H_0: \theta = \theta_0$$
$$H_1: \theta \ne \theta_0$$

(1) $W = \sqrt{n} \times \sqrt{I(\theta)}(\hat{\theta}-\theta) \sim N(0,1)$ or $W = n \times I(\theta)(\hat{\theta}-\theta) \sim \chi^2(1)$

where $I(\theta)$ is the Fisher Information of $\theta$. Normally we can plug in $I(\hat{\theta})$ into the expression. 
 
(2) $W =  \dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})}\sim  N(0,1)$ or $W = \dfrac{(\hat{\theta}-\theta_0)^2}{var(\hat{\theta})} \sim \chi^2(1)$


## Part 2 - Examples with Bernoulli distribution 

In the following hypothesis test, $X \sim Ber(p)$ where $p$ is unknown. 
$$H_0: p = p_0$$
$$H_1: p \ne p_0$$
The Wald test statistic in a given sample is 

$$W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}$$
where $\overline{X}$ is the sample mean. 

[See derivation here](https://github.com/houzhj/Statistics/blob/main/Math/02_f_w_t_t_01.md).

However, when n is small (for example 2 or 3), this statistic does not work well. 

## 2.1 - Very Small Sample Size 
### When the sample size is 2
- Case 1: $\overline{X} = 0 \rightarrow X_1=0$ and $X_2=0$,
  - $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}=\dfrac{\sqrt{2}(0-p_0)}{\sqrt{0(1-0))}}=\infty$.
  - Prob(Case 1) = $(1-p)^2$

- Case 2: $\overline{X} = 1 \rightarrow X_1=1$ and $X_2=1$,
  - $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}=\dfrac{\sqrt{2}(1-p_0)}{\sqrt{1(1-1))}}=\infty$.
  - Prob(Case 2) = $p^2$
 
- Case 3: $\overline{X} = 0.5 \rightarrow X_1=1$ and $X_2=0$, or $X_1=0$ and $X_2=1$,
  - $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}=\dfrac{\sqrt{2}(0.5-p_0)}{\sqrt{0.5(1-0.5))}}=2\sqrt{2}(0.5-p_0)$.
  - Prob(Case 3) = $2p(1-p)$  

It can be seen that the probability of Type 1 error is high (>50\%) if we use the Wald statistics, regardless of the significant level and $p_0$: 
$$P(W=\infty |p=p_0) = (1-p_0)^2+p_0^2 \geq 0.5$$

### When the sample size is 3
- Case 1: $\overline{X} = 0 \rightarrow X_1=0=X_2=X_3=0$,
  - $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}=\dfrac{\sqrt{3}(0-p_0)}{\sqrt{0(1-0))}}=\infty$.
  - Prob(Case 1) = $(1-p)^3$

- Case 2: $\overline{X} = 1 \rightarrow X_1=0=X_2=X_3=1$,
  - $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}=\dfrac{\sqrt{3}(1-p_0)}{\sqrt{1(1-1))}}=\infty$.
  - Prob(Case 2) = $p^3$
 
- Case 3: $\overline{X} \in (0,1)$
  - $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}$.
  - Prob(Case 3) = $1-p^3-(1-p)^3$  

It can be seen that the probability of Type 1 error is high (>25\%) if we use the Wald statistics, regardless of the significant level and $p_0$: 
$$P(W=\infty |p=p_0) = (1-p_0)^3+p_0^3 \geq 0.25$$.


## 2.2 - Simulation Analysis for Wald Tests in Bernoulli distribution 

Consider a $Ber(0.4)$ distribution. The true parameter is $p=0.4$, which is unknown. 

The following codes conduct experiments for Wald test applied in Bernoulli distribution. 
- There are two types of hypothesis tests
  - One-sided:
    $$H_0: \theta \geq \theta_0, H_1: \theta<\theta_0$$
    $$or$$
    $$H_0: \theta \leq \theta_0, H_1: \theta > \theta_0$$
  - Two-sided:
    $$H_0: \theta=\theta_0, H_1: \theta \ne \theta_0$$
- It conducts 1000 experiments. In each experiment, we draw a sample with pre-specified "sample size". Then calculate the $W$ statistic using each of these samples.
- The functions only calculate the test statistics, but not return "reject" or "do not reject" decision. 
- These functions will be used many times in this analysis. We will change the following two settings:
  - The values of $p_0$ in $H_0$. If $p_0 = 0.4$, $H_0$ is true, and rejections indicate Type 1 error; If $p_0 \ne 0.4$, $H_0$ is false, and rejections indicate Type 2 error.
  - Sample size. 
  

```python
p_true     = 0.4
population = np.random.binomial(n=1,p=p_true,size=1000000)
h0         = p_true

def wald_ber_one_side(sample_size,h0):
    dif          = [np.nan]*n_experiment
    var_hat      = [np.nan]*n_experiment
    w            = [np.nan]*n_experiment
    for i in range(n_experiment):
        sample      = np.random.choice(population,size=sample_size,replace=True)
        sample_mean = np.mean(sample)
        dif[i]      = sample_mean-h0
        var_hat[i]  = sample_mean*(1-sample_mean)/sample_size
        if var_hat[i]>0:
            w[i]        = dif[i]/np.sqrt(var_hat[i])
        else:
            w[i] = 99999
    return(w)

def wald_ber_two_side(sample_size,h0):
    dif          = [np.nan]*n_experiment
    var_hat      = [np.nan]*n_experiment
    w            = [np.nan]*n_experiment
    for i in range(n_experiment):
        sample      = np.random.choice(population,size=sample_size,replace=True)
        sample_mean = np.mean(sample)
        dif[i]      = sample_mean-h0
        var_hat[i]  = sample_mean*(1-sample_mean)/sample_size
        if var_hat[i]>0:
            w[i]        = abs(dif[i]/np.sqrt(var_hat[i]))
        else:
            w[i] = 99999
    return(w)

```
An example of how the results look like
```python
wald_ber_two_side(sample_size=20,h0=0.4)
```
The output is a list containing 1000 numbers (only a small portion of the results are shown below), each number is the $W$ statistic in one of 1000 experiments.
|    W Statistis      |
|:-------------------:|
| 2.2360679774997894, |
| 1.3483997249264843, |
|        ......       |
|  0.4494665749754946 |
|  1.549193338482967  |

### Experiment A. Very small sample size
[The codes for this experiment is here]()

Three sample size are considered in the simulation analysis: 2,5,and 30. 

#### One-sided tests
Now, the null hypothesis is $H_0 \leq 0.4$ in all cases. 

(If we use the null hypothesis $H_0: \geq 0.4$, the test statistic would be the same, while the refection region will be different.)

The codes below create test statistics from 1000 experiments. The results show that 
- When the sample size are very small (like 2 and 5), some of the observed $W$ statistics are infinity (because the observed mean is 1 or 0, and therefore the denominater is zero). In these cases, the $W$ statistics are capped at 99999, so that they can be shown in histograms.
- When the sample size is 30, the $W$ statistics exhibit follows a distribution that is close to $N(0,1)$.

```python
h0 = p_true
for s in [2,5,30]:
    sample_size = s
    w           = wald_ber_one_side(sample_size,h0)
    plt.hist(w,bins=10,alpha=0.3,density=True,label='W Statistics')
    plt.title('Wald Statistics, sample size='+str(sample_size))
    plt.legend()
    plt.show()
```

#### Two-sided tests
Now, the null hypothesis is $H_0 = 0.4$. Again, when the sample size is 2 or 5, the distribution of the $W$ statistic is not appropriate for testing. When the sample size
increases to 30, the $W$ statistics exhibit a distribution that is close to a [half-normal](https://en.wikipedia.org/wiki/Half-normal_distribution) distribution. 

```python
for s in [2,5,30]:
    sample_size = s
    w           = wald_ber_two_side(sample_size,h0)
    plt.hist(w,bins=10,alpha=0.3,density=True,label='W Statistics')
    plt.title('Wald Statistics, sample size='+str(sample_size))
    plt.legend()
    plt.show()
```

### Experiment B. Changing null hypothesis with fixed sample size(n=50)
[The codes for this experiment is here]()

Next assume we have a reasonable sample size, say 50. In this experiment, we consider different values of $p_0$. 

Assume the significant level is 0.05 in all cases. The "reject" or "do not reject" decision is according to the following reject regions and p-values, which depend on the type of tests (one-sided or two-sided).

**For two-sided test $H_0: \theta=\theta_0, H_1: \theta \ne \theta_0$**
- Reject if $|W^{obs}|>q_{\alpha/2}$
- p-value = $P(|W| >|W^{obs}|)$

**For one-sided test $H_0: \theta \leq \theta_0, H_1: \theta > \theta_0$**
- Reject if $W^{obs}>q_{\alpha}$
- p-value = $P(W > W^{obs})$


**For one-sided test $H_0: \theta \geq \theta_0, H_1: \theta < \theta_0$**
- Reject if $W^{obs}<-q_{\alpha}$
- p-value = $P(W < W^{obs})$

Note that for one-sided tests, $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}$; and for two-sided tests, $W = \left| \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}\right|$

#### One-sided tests
The code below obtain $W$ statistics through 1000 experiments for each of the followng one-sided tests
- True value: $p=0.4$
- $H_0: p\leq 0.2, H_1: p > 0.2$ (the null hypothesis $H_0$ is false, 1 - rejection rate refelects Type 2 error)
- $H_0: p\leq 0.3, H_1: p > 0.3$ (the null hypothesis $H_0$ is false, 1 - rejection rate refelects Type 2 error)
- $H_0: p\leq 0.4, H_1: p > 0.4$ (the null hypothesis $H_0$ is true, rejection rate refelects Type 1 error)
- $H_0: p\leq 0.5, H_1: p > 0.5$ (the null hypothesis $H_0$ is true, rejection rate refelects Type 1 error)
- $H_0: p\leq 0.6, H_1: p > 0.6$ (the null hypothesis $H_0$ is true, rejection rate refelects Type 1 error)


```python
alpha          = 0.05
critical_value = norm.ppf(1-alpha)
h0_list        =  [0.2,0.3,0.4,0.5,0.6]
summary_table  = pd.DataFrame(columns=['h0','Reject_Rate','Error'])

for h in range(len(h0_list)):
    w = wald_ber_one_side(50,h0_list[h])
    reject = [int(w[i]>=critical_value) for i in range(len(w))]
    summary_table.loc[h,'h0'] = 'p<='+str(h0_list[h])
    summary_table.loc[h,'Reject_Rate'] = "{:.3f}".format(np.mean(reject))
    if (h0_list[h]<p_true):
        summary_table.loc[h,'Error'] = 'Type 2: ' + str("{:.3f}".format(1-np.mean(reject)))
    else:
        summary_table.loc[h,'Error'] = 'Type 1: ' + str("{:.3f}".format(np.mean(reject)))
    plt.hist(w,bins=10,alpha=0.8,density=False,label='W Statistics')
    plt.axvline(x=critical_value, color='green', linestyle='dashed', linewidth=2, label='critical_value')
    if critical_value < min(w):
        plt.axvspan(critical_value,max(w),facecolor='red', alpha=0.1)
    if critical_value > max(w):
        plt.axvspan(min(w),critical_value,facecolor='green', alpha=0.1)
    if (critical_value > min(w)) & (critical_value < max(w)):
        plt.axvspan(min(w), critical_value, facecolor='green', alpha=0.1)
        plt.axvspan(critical_value, max(w), facecolor='red', alpha=0.1)
    plt.title('Wald Statistics, null hypothesis: p<='+str(h0_list[h])+', Reject if W>CV (red area)')
    plt.legend()
    plt.show()
```
|     h0 | Reject_Rate |         Error |
|-------:|------------:|--------------:|
| p<=0.2 |       0.898 | Type 2: 0.102 |
| p<=0.3 |       0.438 | Type 2: 0.562 |
| p<=0.4 |       0.057 | Type 1: 0.057 |
| p<=0.5 |       0.001 | Type 1: 0.001 |
| p<=0.6 |       0.000 | Type 1: 0.000 |

#### Two-sided tests
The code below obtain $W$ statistics through 1000 experiments for each of the followng two-sided tests
- True value: $p=0.4$
- $H_0: p = 0.2, H_1: p \ne 0.2$ (the null hypothesis $H_0$ is false, 1 - rejection rate refelects Type 2 error)
- $H_0: p = 0.3, H_1: p \ne 0.3$ (the null hypothesis $H_0$ is false, 1 - rejection rate refelects Type 2 error)
- $H_0: p = 0.4, H_1: p \ne 0.4$ (the null hypothesis $H_0$ is true, rejection rate refelects Type 1 error)
- $H_0: p = 0.5, H_1: p \ne 0.5$ (the null hypothesis $H_0$ is false, 1 - rejection rate refelects Type 2 error)
- $H_0: p = 0.6, H_1: p \ne 0.6$ (the null hypothesis $H_0$ is false, 1 - rejection rate refelects Type 2 error)

The codes are similar. The main differences are now the test statistic is an absolute value and the critical values is based on the $\alpha/2$ quantile, instead of $\alpha$. 
```python
alpha          = 0.05
critical_value = norm.ppf(1-alpha/2)
h0_list        =  [0.2,0.3,0.4,0.5,0.6]
summary_table = pd.DataFrame(columns=['h0','Reject_Rate','Error'])
for h in range(len(h0_list)):
    w = wald_ber_two_side(50,h0_list[h])
    reject = [int(w[i]>=critical_value) for i in range(len(w))]
    summary_table.loc[h,'h0'] = 'p='+str(h0_list[h])
    summary_table.loc[h,'Reject_Rate'] = "{:.3f}".format(np.mean(reject))
    if (h0_list[h]!=p_true):
        summary_table.loc[h,'Error'] = 'Type 2: ' + str("{:.3f}".format(1-np.mean(reject)))
    else:
        summary_table.loc[h,'Error'] = 'Type 1: ' + str("{:.3f}".format(np.mean(reject)))
    plt.hist(w,bins=10,alpha=0.8,density=False,label='W Statistics')
    plt.axvline(x=critical_value, color='green', linestyle='dashed', linewidth=2, label='critical_value')
    if critical_value < min(w):
        plt.axvspan(critical_value,max(w),facecolor='red', alpha=0.1)
    if critical_value > max(w):
        plt.axvspan(min(w),critical_value,facecolor='green', alpha=0.1)
    if (critical_value > min(w)) & (critical_value < max(w)):
        plt.axvspan(min(w), critical_value, facecolor='green', alpha=0.1)
        plt.axvspan(critical_value, max(w), facecolor='red', alpha=0.1)
    plt.title('Wald Statistics, null hypothesis: p='+str(h0_list[h])+', Reject if W>CV (red area)')
    plt.legend()
    plt.show()
```

|     h0 | Reject_Rate |         Error |
|-------:|------------:|--------------:|
| p<=0.2 |       0.898 | Type 2: 0.102 |
| p<=0.3 |       0.438 | Type 2: 0.562 |
| p<=0.4 |       0.057 | Type 1: 0.057 |
| p<=0.5 |       0.001 | Type 1: 0.001 |
| p<=0.6 |       0.000 | Type 1: 0.000 |


### Experiment C. Changing the sample size
[The codes for this experiment is here]()

Now consider the impact of sample size. 
- We fixed the significant level of the test at 5\%.
- Consider 4 choices of sample size: 30, 50, 100, 500
- Consider 2 two-sided tests:
  - $H_0: p=0.4, H_1: p \ne 0.4$. In this case, the null hypothesis is true. Rejection leads to Type 1 error.
  - $H_0: p=0.5, H_1: p \ne 0.5$. In this case, the null hypothesis is false. Failing to reject leads to Type 2 error. 

The following codes conduct the simulation analysis for $H_0: p=0.4, H_1: p \ne 0.4$. 
```python
alpha            = 0.05
critical_value   = norm.ppf(1-alpha/2)
sample_size_list = [30, 50,100,500]
summary_table = pd.DataFrame(columns=['sample_size','Reject_Rate','Error'])
for s in range(len(sample_size_list)):
    w = wald_ber_two_side(sample_size_list[s],0.4)
    reject = [int(w[i]>=critical_value) for i in range(len(w))]
    summary_table.loc[s,'sample_size'] = str(sample_size_list[s])
    summary_table.loc[s,'Reject_Rate'] = "{:.3f}".format(np.mean(reject))
    summary_table.loc[s,'Error'] = 'Type 1: ' + str(round(np.mean(reject),4))
    plt.hist(w,bins=10,alpha=0.8,density=False,label='W Statistics')
    plt.axvline(x=critical_value, color='green', linestyle='dashed', linewidth=2, label='critical_value')
    if critical_value < min(w):
        plt.axvspan(critical_value,max(w),facecolor='red', alpha=0.1)
    if critical_value > max(w):
        plt.axvspan(min(w),critical_value,facecolor='green', alpha=0.1)
    if (critical_value > min(w)) & (critical_value < max(w)):
        plt.axvspan(min(w), critical_value, facecolor='green', alpha=0.1)
        plt.axvspan(critical_value, max(w), facecolor='red', alpha=0.1)
    plt.title('Wald Statistics, null hypothesis: p=0.4,'+' Sample Size:'+str(sample_size_list[s])+', Reject if W>CV (red area)')
    plt.legend()
    plt.show()
summary_table
```
The results shows that increasing the sample size from 30 to 500 does not significantly reduct the probability of Type 1 error.

| sample_size | Reject_Rate |         Error |
|------------:|------------:|--------------:|
|          30 |       0.078 | Type 1: 0.078 |
|          50 |       0.049 | Type 1: 0.049 |
|         100 |       0.059 | Type 1: 0.059 |
|         500 |       0.054 | Type 1: 0.054 |

The following codes conduct the simulation analysis for $H_0: p=0.5, H_1: p \ne 0.5$. 

```python
alpha            = 0.05
critical_value   = norm.ppf(1-alpha/2)
sample_size_list = [30, 50,100,500]
summary_table = pd.DataFrame(columns=['sample_size','Reject_Rate','Error'])
for s in range(len(sample_size_list)):
    w = wald_ber_two_side(sample_size_list[s],0.5)
    reject = [int(w[i]>=critical_value) for i in range(len(w))]
    summary_table.loc[s,'sample_size'] = str(sample_size_list[s])
    summary_table.loc[s,'Reject_Rate'] = "{:.3f}".format(np.mean(reject))
    summary_table.loc[s,'Error'] = 'Type 2: ' + str("{:.3f}".format(1-np.mean(reject)))
    plt.hist(w,bins=10,alpha=0.8,density=False,label='W Statistics')
    plt.axvline(x=critical_value, color='green', linestyle='dashed', linewidth=2, label='critical_value')
    if critical_value < min(w):
        plt.axvspan(critical_value,max(w),facecolor='red', alpha=0.1)
    if critical_value > max(w):
        plt.axvspan(min(w),critical_value,facecolor='green', alpha=0.1)
    if (critical_value > min(w)) & (critical_value < max(w)):
        plt.axvspan(min(w), critical_value, facecolor='green', alpha=0.1)
        plt.axvspan(critical_value, max(w), facecolor='red', alpha=0.1)
    plt.title('Wald Statistics, null hypothesis: p=0.5,'+' Sample Size:'+str(sample_size_list[s])+', Reject if W>CV (red area)')
    plt.legend()
    plt.show()
summary_table
```
The results shows that increasing sample size (from 30 to 500) improves the power of a test.
| sample_size | Reject_Rate |         Error |
|------------:|------------:|--------------:|
|          30 |       0.164 | Type 2: 0.836 |
|          50 |       0.350 | Type 2: 0.650 |
|         100 |       0.553 | Type 2: 0.447 |
|         500 |       0.997 | Type 2: 0.003 |

If we consider a null hypothesis using a value more close to but not equal to 0.4, a larger sample size will be needed to get a test with high power.
Consider the test $H_0: p = 0.41$.

```python
alpha            = 0.05
critical_value   = norm.ppf(1-alpha/2)
sample_size_list = [30,50,100,500,5000,10000,20000,30000]
summary_table = pd.DataFrame(columns=['sample_size','Reject_Rate','Error'])
for s in range(len(sample_size_list)):
    w = wald_ber_two_side(sample_size_list[s],0.41)
    reject = [int(w[i]>=critical_value) for i in range(len(w))]
    summary_table.loc[s,'sample_size'] = str(sample_size_list[s])
    summary_table.loc[s,'Reject_Rate'] = "{:.3f}".format(np.mean(reject))
    summary_table.loc[s,'Error'] = 'Type 2: ' + str("{:.3f}".format(1-np.mean(reject)))
summary_table
```

As shown in the table below, the Type 2 error rate is higher than 10\% until the sample size increases to 30000.

| sample_size | Reject_Rate |     Error     |
|:-----------:|:-----------:|:-------------:|
|      30     |    0.069    | Type 2: 0.931 |
|      50     |    0.072    | Type 2: 0.928 |
|     100     |    0.056    | Type 2: 0.944 |
|     500     |    0.069    | Type 2: 0.931 |
|     5000    |    0.259    | Type 2: 0.741 |
|    10000    |    0.499    | Type 2: 0.501 |
|    20000    |    0.806    | Type 2: 0.194 |
|    30000    |    0.939    | Type 2: 0.061 |


## Part 3 - Examples with Normal distribution 
Consider the Wald test for the mean of a $N(\mu,\sigma^2)$ distribution. The $W$ statistic in a given sample is calculated as 
$$W =  \dfrac{\hat{\mu}-\mu}{\sqrt{\widehat{var}(\hat{\mu})}} = \dfrac{\hat{\mu}-\mu}{\sqrt{s_n^2/n}} = \dfrac{\sqrt{n}(\hat{\mu}-\mu)}{\sqrt{s_n^2/n}}$$
where 
$$s_n^2 = \dfrac{1}{n-1} \sum_{i=1}^n \left( X_i-\overline{X}\right)^2$$

Under $H_0: \mu = \mu_0$, and $n$ is large enough 
$$W \rightarrow \dfrac{\sqrt{n}(\overline{X}-\mu)}{\sigma} \sim N(0,1)$$

[see derivations here](https://github.com/houzhj/Statistics/blob/main/Math/02_f_w_t_t_02.md)














