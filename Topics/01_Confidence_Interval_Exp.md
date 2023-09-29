## Example B - Exponential Distribution

### Part 1 - Math Derivations

$$X \sim Exp\left(\lambda \right), \hat{\lambda} = \dfrac{1}{\overline{X}}$$

By central limit theorem and delta method, $\sqrt{n}\left(\dfrac{1}{\overline{X}}-\lambda \right) \rightarrow N(0,\lambda^2)$. 

The confidence interval can be written as 
$$\dfrac{1}{\overline{X}} \pm q_{1-\alpha/2} \dfrac{\sqrt{\lambda^2)}}{\sqrt{n}} = \dfrac{1}{\overline{X}} \pm q_{1-\alpha/2} \dfrac{\lambda}{\sqrt{n}}$$

1. **Conservative Bound**
   
   $\lambda>0$ is not bounded, so 
   $$CI_{cons} = (-\infty,\infty)$$
   
2. **Solve**

   According to the following derivations
   $$\hat{\lambda} - q_{1-\alpha/2} \dfrac{\lambda}{\sqrt{n}} \leq  \lambda \leq \hat{\lambda} + q_{1-\alpha/2} \dfrac{\lambda}{\sqrt{n}}$$
   $$\downarrow$$
   $$\lambda \geq \hat{\lambda} \left( 1+ \dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1},\lambda \leq \hat{\lambda} \left(1-\dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1}$$
   $$\downarrow$$
   The confidence interval is
   $$CI_{solve} = \left (\hat{\lambda} \left( 1+ \dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1}, \hat{\lambda} \left(1-\dfrac{q_{1+\alpha/2}}{\sqrt{n}} \right)^{-1} \right)
   = \left (\dfrac{1}{\overline{X}} \left( 1+ \dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)^{-1}, \dfrac{1}{\overline{X}}\left(1-\dfrac{q_{1+\alpha/2}}{\sqrt{n}} \right)^{-1} \right)$$

4. **Plug-in**
   Given that $\hat{\lambda}=\dfrac{1}{\overline{X}}$, the confidence interval is
   $$CI_{plug-in} = \dfrac{1}{\overline{X}} \pm q_{1-\alpha/2} \dfrac{\hat{\lambda}}{\sqrt{n}}=\left(
   \dfrac{1}{\overline{X}} \left(1-\dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right) , \dfrac{1}{\overline{X}} \left(1+\dfrac{q_{1-\alpha/2}}{\sqrt{n}} \right)\right)$$


### Part 2 - Python codes

The code is [here]().

Then calculate the three types of confidence intervals using simulated data. Assume the true value of $\lambda$ is 3. Consider a level 90% for Confidence Intervals (i.e. $\alpha$=10%). The codes are similar with those for Example A.

```python
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
