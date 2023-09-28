# Hypothesis Testing Example with a Uniform Distribution
### ipynb file can be found [here](https://github.com/houzhj/Statistics/blob/main/ipynb/02_testing_with_uniform_distribution.ipynb)

Consider a hypothesis testing with a random variable $X \sim U[a,b]$, with 
$$H_0: b = c$$
$$H_1: b \leq c$$
where $c$ is a given constant. Also, let's assume the lower bound parameter $a$ is known, so in this case the test is mainly about the upper bound parameter $b$. 
Assume we have $n$ observations in total
$$X_1,…,X_n$$ 
or 
$$X_{(1)} \leq X_{(2)} \leq ... \leq X_{(n)}$$
where $X_{(i)}$ is the $i^{th}$ largest number among $X_1,...,X_n$

Clearly, $a \leq X_{(1)} \leq X_{(n)} \leq c$. If any observed $X_i>c$, we can directly conclude that the null hypothesis is true. So let assume 
$$X_{(1)} \leq ... \leq X_{(n)} \leq c$$

As discussed [here](https://github.com/houzhj/Statistics/edit/main/02_Wald_LR_Tests.md), the Wald Test and the Likelihood Ratio Test (both are maximum-likelihood-based tests) can not be directly applied for uniform distribution. The purpose of this study is to investegate alternative testing methods. 

## 1. Test Statistic
The maximum likelihood estimator of the upper bound parameters $b$ can be derived as below. 

The likelihood function
$$L(a,b) = \prod_{i=1}^{n} \dfrac{1}{b-a} = (b-a)^{-n}$$
$$\downarrow$$
The log-likelihood function
$$l(a,b) = -n \times ln(b-a)$$
$$\downarrow$$
Take the derivative with respect to $b$
$$\dfrac{\partial l(a,b)}{\partial b} = \dfrac{-n}{b-a}<0$$

So $L(a,b)$ is monotonically decreasing with respect to $b$, and $L(a,b)$ is maximized at the smallest possible value of $b$. 
According to observations $(X_1,...,X_n)$,
$$\hat{b}= max(X_1,...,X_n)=X_{(n)}$$

So any test we could do might involve the test statistic $\hat{b}=X_{(n)}$. Let's just use $X_((n))$ as the test statistic.

$$T_n = X_((n))$$ 

## 2. Rejection Region
The reject region $R$ is defined so that the decision is to reject $H_0$ if $T_n = X_{(n)} \in R$. Consider the rejection region defined by a scalar $r$, reject the null hypothesis $H_0: b=c$ if
$$T_n \in [a,c]$$

Note that we must have $a < r < c$, otherwise the test is logical meaningless. 
- If we chose $r < a$: it is impossible that the maximum of $X_1,...,X_n$ is smaller than a number that is smaller than the lower bound $a$. I.e., the null hypothesis will always be rejected. 
- If we chose $r > c$: the null hypothesis will be rejected if we observed $X_{(n)}$ that is greated than $r$, which is greater than $c$. In this case we can conclude that the null hypothesis is true withtout any statistical testing, and the test is essentially unnecessary.  

## 3. Type 1 and Type 2 Error
The probability of Type 1 and Type 2 Error of this tests can be expressed by functions of known variables ($a,r,c,n$). In specific
- $a$: the lower bound of the uniform distribution
- $r$: the scalar that defines the reject region, reject the $H_0$ if $T_n = X_{(n)} \in [a,r]$
- $c$: the value in the null hypothesis, $H_0: b=c$, $H_1: b < c$
- $n$: the number of observations in the sample

According to the [definition](https://en.wikipedia.org/wiki/Power_of_a_test) to Type 1 (and significant level) and Type 2 errors (and power), 

- The significant level of this test is the probability of rejecting $H_0$ ($T_{(n)} \in [a,r]$) when $H_0$ is true ($b \geq c$).

$$\alpha = P(a < T_n < r) | b \geq c) = P(a < X_{(n)} < r) | b \geq c) $$

$$ =  P[ (a < X_1 < r) | b \geq c) \cap (a < X_2 < r) | b \geq c) \cap ... \cap (a < X_n < r) | b \geq c)]$$ 

$$ = \left(\dfrac{r-a}{c-a} \right)^n$$

- The power of this test is the 1 minus probability of not rejecting $H_0$ ($T_{(n)} \notin [a,r]$) when $H_0$ is false ($b < c$).

$$ 1-\beta = 1-P(T_n \geq r | b < c) = 1-[1-P(a < T_n < r | b < c)] = P(a <  X_{(n)} < r | b < c)$$ 

$$ = P[ (a < X_1 < r) | b < c) \cap (a < X_2 < r) | b < c) \cap ... \cap (a < X_n < r) | b < c)]$$

$$\geq P[ (a < X_1 < r) | b = c) \cap (a < X_2 < r) | b = c) \cap ... \cap (a < X_n < r) | b = c)]$$ 

$$ =  \left(\dfrac{r-a}{c-a} \right)^n$$

## 4. Applications
We can use these conclusions to 
- calculate the significant level of a test, or
- design a test(i.e., finding the rejection region) to achieve a given significant level. 

Assume we have the following information:

(1) The random variable $X \sim U(0,b)$, where $b$ is the unknown upper bound, and is the parameter of interest.

(2) We have $n=10$ observations $X_1, ..., X_{10}$. The sample size can be any positive integer, and we fix it at 10. 

(3) We observe the maximum value among $X_1, ..., X_{10}$. Then we have the hypotheses $H_0: b = c$, $H_1: b < c$. Note that we will only use $c > max(X_1,...,X_{10})$ in the hypotheses, otherwise the test will be meaningless - no one needs to test if $b=4$ after observing the sample maximum at 5. Instead, one might be interested in whether the upper bound parameter is 6.  

### Calculating the significant level of a given test 
Let's say the lower bound parameter is given: $a=5$. We conduct a test with pre-specified rejection region $[5,9]$, and hypotheses $H_0: b=10$, $H_1: b < 10$,. 

If the maximum in the sample $(n=8)$ is greater then 10, we can direcly conclude that $b>10$ and there is no need to conduct this test. 

Otherwise, we will reject $H_0: b=10$ if $X_{(10)}<9$. 

In this test, we have $a=5$, $n=8$, $c=10$, $r=9$, so the significant level is 
$$\left(\dfrac{r-a}{c-a} \right)^n = \left(\dfrac{9-5}{10-5} \right)^8 = 0.8^8 = 0.167772$$

The following codes calculate the significant level with the following information 
- $a=5$(known), and $b=10$ (unknown). Since the significant level is for Type 1 error, the actual value of the parameter is the same as the value in $H_0$. That is, $H_0: b=10$
- Reject the $H_0$ is $X_{(n)} < r$. Ten values of $r$ are considered: $9.0, 9.1, ..., 9.9.
- Four sample sizes are considered: 10, 20, 50, 100. As the sample size increases, the maximum observable sample value tends to be larger.

```python
a      = 5
h0     = 10
true_b = 10
r_list = [9+0.1*i for i in range(10)]
sample_size_list = [10,20,50,100]
```

The following code conducted 100 experiment for all the combinations of $r$ and $n$. In each combination, the rejection rate (Type 1 Error rate) is calculated. This is the simulation-based significant level. 
```python
alpha_n_r_sim = pd.DataFrame(columns = ['r values']+['n='+str(i) for i in sample_size_list])
for r in range(len(r_list)):
    alpha_n_r_sim.loc[r,'r values'] = r_list[r]
    for ss in range(len(sample_size_list)):
        n_experiment = 100
        result       = pd.DataFrame(columns=['ts','reject'])
        for e in range(n_experiment):
            sample = np.random.uniform(a,true_b,sample_size_list[ss])
            ts     = sample.max()
            result.loc[e,'ts']     = ts
            result.loc[e,'reject'] = int(ts<r_list[r])
        alpha_n_r_sim.iloc[r,ss+1] = result['reject'].mean()
alpha_n_r_sim
```

The code below calculate the analytical significant level for each combination of $r$ and $n$, based on $\alpha = \left(\dfrac{r-a}{c-a} \right)^n$ as shown above, 
```python
alpha_n_r_analytical = pd.DataFrame(columns = ['r values']+['n='+str(i) for i in sample_size_list])
for r in range(len(r_list)):
    alpha_n_r_analytical.loc[r,'r values'] = r_list[r]
    for ss in range(len(sample_size_list)):
        alpha_n_r_analytical.iloc[r,ss+1] = ((r_list[r]-a)/(h0-a))**sample_size_list[ss]
alpha_n_r_analytical
```

The simulation-based and the analytical significant levels are compared. In general they are comparable. There are gaps between the numbers in the two tables, probably becasue the number of experiments are not large enough.

<img width="237" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/c9a835b6-9137-466d-a813-1d1042b16251">
<img width="326" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/667029d2-8cc3-4f82-b129-e6321e62c1bb">

### Finding the rejection region with a target significant level

Again the lower bound parameter is given: $a=5$. We have hypotheses $H_0: b=10$, $H_1: b < 10$, and the target significant level is 5%. The sample includes 8 observations.

If the maximum in the sample $(n=8)$ is greater then 10, we can direcly conclude that $b>10$ and there is no need to conduct this test. 

Otherwise, we will need to design a test, specifically to find a value of $r$, and reject $H_0: b=10$ if $X_{(10)} < r$.  

In this test, we have $a=5$, $n=8$, $c=10$, so the significant level is 
$$\alpha = \left(\dfrac{r-a}{c-a} \right)^n = \left(\dfrac{r-5}{10-5} \right)^8 = \left(\dfrac{r-5}{5} \right)^8 $$
Solving this equation yields
$$r = \alpha^{1/n} \times (c-a) + a = 0.05^{1/8} \times (10-5) + 5 = 8.43828$$
This is the value of $r$ - we reject $H_0$ if $X_{(n)} < 8.43828$, and the significant level will be 5%. 

The codes below calculate the rejection region to achieve desired level, with the following options:
- Three significant levels are considered: 0.01,0.05,0.1
- Four sample sizes are considered: 8,20,50,100
- The lower bound parameter is given: $a=5% (known)
- The true upper bound parameter $b=10$ (unknown)
- $H_0: b= 10$ (true)

```python
a          = 5
alpha_list = [0.01,0.05,0.1]
sample_size_list = [8,20,50,100]
h0     = 10

r_n_alpha = pd.DataFrame(columns = ['alpha values']+['n='+str(i) for i in sample_size_list])
for alpha in range(len(alpha_list)):
    r_n_alpha.loc[alpha,'alpha values'] = alpha_list[alpha]
    for ss in range(len(sample_size_list)):
        r_n_alpha.iloc[alpha,ss+1] = alpha_list[alpha]**(1/sample_size_list[ss])*(h0-a)+a
r_n_alpha
```

The results are 

<img width="354" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/52c663dc-f384-4a8d-bb67-ce3de1606488">


#### Checking two cases

(1) $n=20, r=9.456255$: according to the table, the expected $\alpha$ is 10%.

The following codes conducted 10 iterations (each containing 1000 experiments), and calculate the simulation-based $\alpha$. 

```python
n_experiment = 1000
n_now = 20
r_now = 9.456255

alpha_check = [np.nan]*10

for i in range(10):
    result       = pd.DataFrame(columns=['ts','reject'])
    for e in range(n_experiment):
        sample = np.random.uniform(a,true_b,n_now)
        ts     = sample.max()
        result.loc[e,'ts']     = ts
        result.loc[e,'reject'] = int(ts<r_now)
    alpha_check[i]=result['reject'].mean()
alpha_check
```
As shown, all the 10 values are around 10%

<img width="593" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/82256abb-d00a-46de-b79d-e266bf26d875">



(2) $n=50, r=9.560054$: according to the table, the expected $\alpha$ is 1%.

The following codes conducted 10 iterations (each containing 1000 experiments), and calculate the simulation-based $\alpha$. 

```python
n_experiment = 1000
n_now = 50
r_now = 9.560054

alpha_check = [np.nan]*10

for i in range(10):
    result       = pd.DataFrame(columns=['ts','reject'])
    for e in range(n_experiment):
        sample = np.random.uniform(a,true_b,n_now)
        ts     = sample.max()
        result.loc[e,'ts']     = ts
        result.loc[e,'reject'] = int(ts<r_now)
    alpha_check[i]=result['reject'].mean()
alpha_check
```
As shown, all the 10 values are around 1%

<img width="564" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/983e30d5-603c-4dcd-b1c2-98deb153aa70">


