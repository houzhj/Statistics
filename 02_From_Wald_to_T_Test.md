# From Wald Test to Student's t test (T test)
### ipynb file can be found [here]()

# 1. Wald Test
Wald test is a maximum likelihood estimate based test. It is based on the asymptotic normal approximation to the maximum likelihood estimator.
Two equivalent expression of the Wald test statistics are shown below
Consider the parameter $\theta$. The maximum likelihood of $\theta$ is $\hat{\theta}$
The hypotheses are given below (the hypothesis can be one-sided too)
$$H_0: \theta = \theta_0$$
$$H_1: \theta \ne \theta_0$$

(1) $W = \sqrt{n} \times \sqrt{I(\theta)}(\hat{\theta}-\theta) \sim N(0,1)$ or $W = n \times I(\theta)(\hat{\theta}-\theta) \sim \chi^2(1)$

where $I(\theta)$ is the Fisher Information of $\theta$. Normally we can plug in $I(\hat{\theta})$ into the expression. 
 
(2) $W =  \dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})}\sim  N(0,1)$ or $W = \dfrac{(\hat{\theta}-\theta_0)^2}{var(\hat{\theta})} \sim \chi^2(1)$



It can be shown that these expressions are equivalent. 
That asymptotic normal approximation for the MLE says
$$\sqrt{n}(\hat{\theta}-\theta) \sim N(0,I^{-1}(\theta))$$
$$\downarrow$$
$$var(\sqrt{n} \times \hat{\theta}) = I^{-1}(\theta)$$
$$\downarrow$$
$$var(\hat{\theta}) = \frac{1}{\sqrt{n}} \times I^{-1}(\theta) = \dfrac{1}{n I(\theta)}$$
Plugging this to the first expression yields
$$W = \sqrt{n} \times \sqrt{I(\theta)}(\hat{\theta}-\theta) = \sqrt{n I(\theta)}  \times (\hat{\theta}-\theta) =\sqrt{\dfrac{1}{var(\hat{\theta})}} \times (\hat{\theta}-\theta) = \dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})} $$

Consider two examples below. These examples show that a Wald test does not work well when the sample size is very small. We consider hypothesis testing about the parameters from two distributions. 
- Bernoulli distribution 
- Normal distribution (Student's t test is discussed in this example)
  
This study is based on both analytical derivations and simulations.

# 2. Examples with Bernoulli distribution 

In the following hypothesis test, $X \sim Ber(p)$ where $p$ is unknown. 
$$H_0: p = p_0$$
$$H_1: p \ne p_0$$
The Wald test statistic is 

$$ W = \dfrac{(\hat{p}-p_0)}{se(\hat{p})} = \dfrac{(\hat{p}-p_0)}{se(\hat{p})}$$

The maximum likelihood estimator of $p$ is $\hat{p}=\overline{X}$, so the sample variance of $\hat{p}$ is 
$$\widehat{Var}(\hat{p}) = \widehat{Var}(\overline{X}) = \widehat{Var}\left(\dfrac{1}{n} \sum_{i=1}^n X_i \right) =\dfrac{1}{n^2}\widehat{Var}\left(\sum_{i=1}^n X_i  \right)=\dfrac{1}{n^2}\times n \times \widehat{Var}(X_i) = \dfrac{\hat{p}(1-\hat{p})}{n} = \dfrac{\overline{X}(1-\overline{X})}{n}$$

So the observed $W$ statistic in a given sample is 
$$W = \dfrac{(\hat{p}-p_0)}{\sqrt{\widehat{Var}(\hat{p})}} = \dfrac{\overline{X}-p_0}{\sqrt{\dfrac{\overline{X}(1-\overline{X})}{n}}}=\dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}$$

However, when n is small (for example 2 or 3), this statistic does not hold. 

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

It can be seen that the probability of Type 1 error is high ($>50\%$) if we use the Wald statistics, regardless of the critical value and the value in $H_0$:
$P(W=\infty |p=p_0) = (1-p)^2+p^2 \geq 0.5$.

The plot for $y=p^3+(1-p)^3, p\in(0,1)$

<img width="180" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/b898eab1-d017-4022-ae72-35ed98ef9b1d">

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

It can be seen that the probability of Type 1 error is high ($>25\%$) if we use the Wald statistics, regardless of the critical value and the value in $H_0$: $P(W=\infty |p=p_0) = (1-p)^3+p^3 \geq 0.25$.

The plot for $y=p^3+(1-p)^3, p\in(0,1)$

<img width="195" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/a63270cc-a1e0-4935-bdfa-c26ca369034c">

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
- The functions only calculate the test statistics, but not return "reject" or "do not reject" decision. The decision is according to the following reject region, and p-values, which depend on the type of tests (one-sided or two-sided) 
  <img width="593" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/a9e7b9d4-690d-4a3b-b843-a1054d0f4e8c">

  Note that for one-sided tests, $W = \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}$; for two-sided tests, $W = \left| \dfrac{\sqrt{n}(\overline{X}-p_0)}{\sqrt{\overline{X}(1-\overline{X})}}\right|$
- These functions will be used many times in this analysis. We will change the following settings:
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

<img width="194" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/04f40e67-8b40-4f4c-8707-d5716771a925">

### Experiment A. Very small sample size
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
<img width="350" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/a0f7311f-3c2e-45ac-a7af-1869f7196802">

<img width="352" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/2c9c25d4-e61c-42ea-b4f9-d3b6c0b62049">

<img width="369" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/bb28e725-668d-4b85-b522-287e236bffd4">


#### One-sided tests
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

<img width="353" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/4a52ff29-6d9d-4f32-8639-d476e79e6619">

<img width="352" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/6ca7d7cb-8430-4608-bde9-dd6cafe82e61">

<img width="362" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/2f44ce18-5250-46a1-8b97-691b191404ba">



