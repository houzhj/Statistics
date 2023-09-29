## Example A - Bernoulli Distribution

### Part 1 - Math Derivations
$$X \sim Ber(p)$$
The maximum likelihood of $p$ is
$$\hat{p} = \overline{X}$$

By central limit theorem, $\sqrt{n}(\overline{X}-p) \rightarrow N(0,p(1-p))$. 

The confidence interval of $p$ can be written as 
$$\overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$


#### 1. **Conservative Bound**
   
   Since $\sqrt{p(1-p)}\leq\sqrt{0.5(1-0.5)}=0.5$ when $p \in (0,1)$, we have
   $$CI_{cons} = \overline{X} \pm q_{1-\alpha/2}\dfrac{0.5}{\sqrt{n}}$$
   
#### 2. **Solve**

   According to the following derivations
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

   The confidence interval is 
   $$CI_{solve} = \left( \dfrac{-B \pm \sqrt{B^2-4AC}}{2A} \right)$$

#### 3. **Plug-in**
   Given that $\hat{p}=\overline{X}$, the confidence interval is
   $$CI_{plug-in} = \overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{n}} =\overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{\overline{X}(1-\overline{X})}}{\sqrt{n}}$$


### Part 2 - Python Codes

The code is [here]().

Now, calculate the three types of confidence intervals using simulated data. Assume the true value of p is 0.6 (this information is unknown in reality, so it will not be used in the derivation of confidence interval). Consider a level 90% for Confidence Intervals (i.e. $\alpha$=10%)

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

