
## Example C - Gamma Distribution

### Part 1 - Math Derivations

$$X \sim Gamma(\alpha,1/\alpha), \hat{\alpha} = \sqrt{\overline{X}}$$
Note that there is a simplified setting that $\beta = 1/\alpha$, which is not necessarioy the case. So there is only one unknown parameter.

By central limit theorem and delta method, $\sqrt{n}(\sqrt{\overline{X}}-\alpha) \rightarrow N(0,\alpha/4)$. 

The confidence interval can be written as (let $q=q_{1-\alpha/2}$ to avoid two duplicated $\alpha$)
$$\sqrt{\overline{X}} \pm q \dfrac{\sqrt{\alpha/4}}{\sqrt{n}} = \sqrt{\overline{X}} \pm \dfrac{q\sqrt{\alpha}}{2\sqrt{n}}$$

#### 1. **Conservative Bound**
   
   $\alpha>0$ is not bounded, so 
   $$CI_{cons} = (-\infty,\infty)$$
   
#### 2. **Solve**

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
   The confidence interval is 
   $$CI_{solve} = \left( \dfrac{-B \pm \sqrt{B^2-4AC}}{2A} \right)$$

   
#### 3. **Plug-in**
   
   Given that $\hat{\lambda}=\dfrac{1}{\overline{X}}$, the confidence interval is
   $$CI_{plug-in} = \sqrt{\overline{X}} \pm \dfrac{q \sqrt{\hat{\alpha}}}{2\sqrt{n}} = \sqrt{\overline{X}} \pm \dfrac{q\sqrt{\sqrt{\overline{X}}}}{2\sqrt{n}}$$


### Part 2 - Python Codes
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
