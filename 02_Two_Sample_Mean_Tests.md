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


# 2. Code - comparing the two-sample tests by simulation analysis

The function below performs a two sample mean test, given that
- The two random variables $X$ and $Y$ both follow Normal distribution.
- The True mean of $X$ and $Y$, could be either same (H0 is true, rejection rate implies type 1 error), or different (rejection rate implies type 2 error)
- The true variances of $X$ and $Y$, could be same or different
- The sample size of $X$ and $Y$, could be same or different
- The four two-sample testing methods described above: Wald test, t test 1, t test 2, t test 3 are conducted. 

```python
def two_sample_test(mu_x,sigma2_x,size_x,mu_y,sigma2_y,size_y,method):
    n_experiment   = 1000
    results        = pd.DataFrame(columns=['ts','reject'])
    summary        = pd.DataFrame(columns=['size_x','size_y','h0'])

    for e in range(n_experiment):
        data_x   = np.random.normal(loc=mu_x,scale=np.sqrt(sigma2_x),size=size_x) 
        data_y   = np.random.normal(loc=mu_y,scale=np.sqrt(sigma2_y),size=size_y) 
        if method == 'wald_ts': 
            ts,reject = wald_ts(data_x,data_y)
        if method == 't_ts_1':
            ts,reject = t_ts_1(data_x,data_y)
        if method == 't_ts_2':
            ts,reject = t_ts_2(data_x,data_y)
        if method == 't_ts_3':
            ts,reject = t_ts_3(data_x,data_y)
        results.loc[e,'ts']     = ts
        results.loc[e,'reject'] = reject
    
    summary.loc[0,'size_x'] = size_x
    summary.loc[0,'size_y'] = size_y
    if mu_x==mu_y:
        summary.loc[0,'h0']           = 'True'
        summary.loc[0,'type_1_error'] = results['reject'].mean()
    else: 
        summary.loc[0,'h0']           = 'False'
        summary.loc[0,'type_2_error'] = 1- results['reject'].mean()
    
    return results,summary
```

This function works like this. 
```python
two_sample_test(mu_x=4, sigma2_x=3, size_x=40, mu_y=5, sigma2_y=2, size_y=20, method = 't_ts_1')
```

In this example, the true paramaters and the sample sizes for the two random variables are shown as below (they are specified in the function):
$$X \sim N(4,3), n_1 = 40$$

$$Y \sim N(5,2), n_2 = 20$$  

If we conduct the "t_ts_1" method (i.e., t tests with equal sample sizes and variance, see above) 1000 times, we have the following results. Note that in this case, $\mu_1 \ne \mu_2$, so $H_0: \mu_1 = \mu_2$ is false, and failing to reject leads to a Type 2 error. 

<img width="325" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/706ea052-5664-448d-a57d-bdb4f811fb42">

$$$$

The function below conducts a "two_sample_test" using pre-specified settings:
- The function two_sample_test() above is called in this function.
- This function considers two data generation process 
  - In the first test, $\mu_1 = 5$, $\mu_2 = 5$. The null hypothesis is true.
  - In the second test, $\mu_1 = 5$, $\mu_2 = 6$. The null hypothesis is false.
- For simplicity, we do not change the inputs of $\mu_1$ and $\mu_2$, so stick to the settings above. We can change the sample sizes, and variances of $X$ and $Y$. 

```python
def two_sample_test_experiment(sample_size_x_list,sample_size_y_list,
                               sigma_2_x,sigma_2_y,
                               method):
    mu_x      = 5
    mu_y1     = 5
    mu_y2     = 6
    ### H0 is true
    for s in range(len(sample_size_x_list)):
        results_now,summary_now = two_sample_test(mu_x=mu_x, sigma2_x=sigma_2_x,size_x=sample_size_x_list[s],
                                                  mu_y=mu_y1,sigma2_y=sigma_2_y,size_y=sample_size_y_list[s],
                                                  method = method)
        if s==0:
            summary_t = summary_now
        else:
            summary_t = pd.concat([summary_t,summary_now],axis=0)
    ### H0 is false
    for s in range(len(sample_size_x_list)):
        results_now,summary_now = two_sample_test(mu_x=mu_x, sigma2_x=sigma_2_x,size_x=sample_size_x_list[s],
                                                  mu_y=mu_y2,sigma2_y=sigma_2_y,size_y=sample_size_y_list[s],
                                                  method = method)
        if s==0:
            summary_f = summary_now
        else:
            summary_f = pd.concat([summary_f,summary_now],axis=0)
    summary = summary_t.drop(['h0'],axis=1).merge(summary_f.drop(['h0'],axis=1),on=['size_x','size_y'])
    summary['x_var']  = sigma_2_x
    summary['y_var']  = sigma_2_y

    summary['sample size'] = 'Equal' if sample_size_x_list==sample_size_y_list else 'Unequal'
    summary['variance']    = 'Equal' if sigma_2_x==sigma_2_y else 'Unequal'
            
    summary['method'] = method
    return summary
```

The function works like below. 
```python
two_sample_test_experiment(sample_size_x_list=[10,20], sample_size_y_list[10,30],
                           sigma_2_x=2, sigma_2_y=3,
                           method='wald_ts')
```
In this example, we considered two combinations of sample sizes of $X$ and $Y$, represented by the two rows in the output table.
- In the first case, $n_1=10$, $n_2=10$. We further conducted two data generation processes:
  -  $X \sim N(5,2)$, $Y \sim N(5,3)$. In this case, the $H_0$ is true.
  -  $X \sim N(6,2)$, $Y \sim N(5,3)$. In this case, the $H_0$ is false.
- Similarly, in the second case, $n_1=20$, $n_2=30$. Again we further conducted two data generation processes:
  -  $X \sim N(5,2)$, $Y \sim N(5,3)$. In this case, the $H_0$ is true.
  -  $X \sim N(6,2)$, $Y \sim N(5,3)$. In this case, the $H_0$ is false.
Then we conduct the Wald test analysis, using function two_sample_test() for each of the four combinations. 

<img width="568" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/d1f2de93-431e-4153-aef6-f0f4ff41dcc2">

Interpretation of the results
- The first row is for $n_1=10$, $n_2=10$. The second row is for $n_1=20$, $n_2=30$.
- Column 'type_1_error' is the rejection rate among the 1000 experiments, when $H_0$ is true (i.e, when the true means are $\mu_1=\mu_2=5$).  
- Column 'type_2_error' is the rejection rate among the 1000 experiments, when $H_0$ is false (i.e, when the true means are $\mu_1=5, \mu_2=6$).
- Column 'sample size' is 'unequal' becasue the inputs 'sample_size_x_list' and 'sample_size_y_list' are not completely equal to each other, although in one case the sample size are both 10.
- Column 'variance' is 'unequal' becasue 'sigma_2_x'=2, 'sigma_2_y'=3,


# 3. Results of the simulation-based analysis
Consider the following cases
## Case 1: Equal variance, equal sample size
$X \sim N(\mu_1, 2), Y \sim N(\mu_2, 2)$

Sample size: $n_1 = n_2 = [5,10,20,50,100]$

```python
sample_size_list_1 = [5,10,20,50,100]
sample_size_list_2 = [50,50,50,50,50]
sigma_2_x          = 2
sigma_2_y          = 2
wald_result = two_sample_test_experiment(sample_size_list_1,sample_size_list_1,sigma_2_x,sigma_2_y,'wald_ts')
t1_result   = two_sample_test_experiment(sample_size_list_1,sample_size_list_1,sigma_2_x,sigma_2_y,'t_ts_1')
t2_result   = two_sample_test_experiment(sample_size_list_1,sample_size_list_1,sigma_2_x,sigma_2_y,'t_ts_2')
t3_result   = two_sample_test_experiment(sample_size_list_1,sample_size_list_1,sigma_2_x,sigma_2_y,'t_ts_3')
combined_results = pd.concat([wald_result,t1_result,t2_result,t3_result],axis=0)
combined_results['label'] = combined_results['method'].apply(create_label)

plt.figure(figsize=(7,6))
sns.scatterplot(data = combined_results,
                x="type_2_error",y="type_1_error", 
                hue="label", size="size_x")
plt.title('Equal sample size, equal variance')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
```

<img width="783" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/d932ccc7-eb94-4d25-aed8-171309b5390b">



## Case 2: Unequal variance, equal sample size
$X \sim N(\mu_1, 2), Y \sim N(\mu_2, 3)$

Sample size: $n_1 = n_2 = [5,10,20,50,100]$

```python
```



## Case 3: Equal variance, unequal sample size
$X \sim N(\mu_1, 2), Y \sim N(\mu_2, 2)$

Sample size: $n_1 = [5,10,20,50,100]$, $n_2 = [50,50,50,50,50]$

```python
```






## Case 4: Unequal variance, unequal sample size
$X \sim N(\mu_1, 2), Y \sim N(\mu_2, 3)$

Sample size: $n_1 = [5,10,20,50,100]$, $n_2 = [50,50,50,50,50]$

```python
```

















