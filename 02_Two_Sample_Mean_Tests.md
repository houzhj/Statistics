# Two Sample Tests
### ipynb file can be found [here]()

Consider two random variables $X$ and $Y$. 
$$E(X) = \mu_1, Var(X) = \sigma_1^2$$
$$E(Y) = \mu_2, Var(Y) = \sigma_2^2$$

Assume 
- $X_1,...,X_m$  are independent random samples of $X$ 
- $Y_1,...,Y_n$  are independent random samples of $Y$
- The samples from $X$ and from $Y$ are independent

Consider the hypothesis testing
$$H_0: \mu_1 = \mu_2$$
$$H_1: \mu_1 \ne \mu_2$$

## The goal of this study is to conduct and compare several two-sample tests (introduced below) under different scenarios
- Equal/unequal sample size
- Equal/unequal variance
- The true mean of X and Y, could be either same ($H_0$ is true, rejection is related to Type 1 Error), or different ($H_0$ is false, rejection is related to Type 2 error)


# 1. Several two-sample tests in the literature
There are multiple two-sample tests in the literature. The derivations of the test statistics and the distribution of these statistics under the null hypothesis are not discussed. The links and the snapshots are from [the article about Student's t-test in wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test). 

## (1) Wald test
Equivalently, let $\theta = \mu_X - \mu_Y$, the hypothesis test can be rewritten as
$$H_0: \theta = 0 $$
$$H_1: \theta \ne 0 $$
The test statistic is defined by 

$$W = \dfrac{\hat{\theta}-\theta}{\sqrt{\dfrac{\hat{\sigma_1}^2}{m}  + \dfrac{\hat{\sigma_2}^2}{n} }} \sim N(0,1)$$
where

$$\hat{\sigma_1}^2 = \frac{1}{m} \sum_{i=1}^n (X_i-\overline{X})^2$$

$$\hat{\sigma_2}^2 = \frac{1}{n} \sum_{i=1}^n (Y_i-\overline{Y})^2$$

The codes below conduct this test. 

```python
def wald_ts(data_x,data_y):
    alpha  = 0.05
    critical_value = norm.ppf(1-alpha/2)
    x_mean = np.mean(data_x)
    x_var  = np.var(data_x)
    x_s2   = np.var(data_x,ddof=1)    
    size_x = len(data_x)
    y_mean = np.mean(data_y)
    y_var  = np.var(data_y)
    y_s2   = np.var(data_y,ddof=1)
    size_y = len(data_y)
    ts      = abs(x_mean-y_mean)/np.sqrt(x_var/size_x+y_var/size_y)
    reject = int(ts>critical_value)
    return ts,reject
```

## (2) t test 1 - Equal sample sizes and variance
#### [link](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_sample_sizes_and_variance)

#### Snapshot
<img width="962" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/337be7c1-8708-44c7-a7b4-785f15cbca24">

The codes below conduct this test. 
```python
def t_ts_1(data_x,data_y):
    alpha  = 0.05
    critical_value = t.ppf(1-alpha/2, df=2*len(data_x)-2)
    x_mean = np.mean(data_x)
    x_var  = np.var(data_x)
    x_s2   = np.var(data_x,ddof=1)    
    size_x = len(data_x)
    y_mean = np.mean(data_y)
    y_var  = np.var(data_y)
    y_s2   = np.var(data_y,ddof=1)
    size_y = len(data_y)
    sp     = np.sqrt(0.5*(x_s2+y_s2))
    ts     = abs(x_mean-y_mean)/(sp*np.sqrt(2/size_x))
    reject = int(ts>critical_value)
    return ts,reject
```

## (3) t test 2 - Equal or unequal sample sizes, similar variances
#### [link](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_(1/2_%3C_sX1/sX2_%3C_2))

#### Snapshot
<img width="953" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/3bf8f6c9-87f5-4642-b5c2-89e66c7f0b92">

The codes below conduct this test. 
```python
def t_ts_2(data_x,data_y):
    alpha  = 0.05
    critical_value = t.ppf(1-alpha/2, df=2*len(data_x)-2)
    x_mean = np.mean(data_x)
    x_var  = np.var(data_x)
    x_s2   = np.var(data_x,ddof=1)    
    size_x = len(data_x)
    y_mean = np.mean(data_y)
    y_var  = np.var(data_y)
    y_s2   = np.var(data_y,ddof=1)
    size_y = len(data_y)
    sp     = np.sqrt(((size_x-1)*x_s2+(size_y-1)*y_s2)/(size_x+size_y-2))
    ts     = abs(x_mean-y_mean)/(sp*np.sqrt(1/size_x+1/size_y))
    reject = int(ts>critical_value)
    return ts,reject
```

## (4) t test 3 - Equal or unequal sample sizes, unequal variances
#### [link](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_unequal_variances_(sX1_%3E_2sX2_or_sX2_%3E_2sX1))

#### Snapshot
<img width="966" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/b0d8e426-6ed5-40de-8451-f0254d15a361">

The codes below conduct this test. 
```python
def t_ts_3(data_x,data_y):
    alpha          = 0.05
    x_mean = np.mean(data_x)
    x_var  = np.var(data_x)
    x_s2   = np.var(data_x,ddof=1)    
    size_x = len(data_x)
    y_mean = np.mean(data_y)
    y_var  = np.var(data_y)
    y_s2   = np.var(data_y,ddof=1)
    size_y = len(data_y)
    sp     = np.sqrt(((size_x-1)*x_s2+(size_y-1)*y_s2)/(size_x+size_y-2))
    ts     = abs(x_mean-y_mean)/(sp*np.sqrt(1/size_x+1/size_y))
    dof    = (x_s2/size_x+y_s2/size_y)**2/((x_s2/size_x)**2/(size_x-1)+(y_s2/size_y)**2/(size_y-1))
    critical_value = t.ppf(1-alpha/2, df=dof)
    reject = int(ts>critical_value)
    return ts,reject
```


# 2. Comparing the two-sample tests by simulation analysis

The function below performs a two sample mean test, given that
- The two random variables $X$ and $Y$ both follow Normal distribution.
- The True mean of $X$ and $Y$, could be either same (H0 is true, rejection rate implies type 1 error), or different (rejection rate implies type 2 error)
- The true variances of $X$ and $Y$, could be same or different
- The sample size of $X$ and $Y$, could be same or different
- The four two-sample testing methods described above: Wald test, t test 1, t test 2, t test 3 are conducted. 









