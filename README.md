# Statistics
These studies are about various topics in statistics, including mathematical derivations of key statistical conclusions and their applications, implementation using Python code, and case studies based on simulations.

## 1. Some important concepts in statistics
- Central Limit Theorem
- Distribution
- [Estimations and Confidence Interval](https://github.com/houzhj/Statistics/blob/main/ipynb/01_confidence_intervals.ipynb)
  - Three methods to calculate the confidence interval of a parameter: Conservative**, Solve, Plug-in.
  - **Key words**: confidence interval
- [Delta Method](https://github.com/houzhj/Statistics/blob/main/ipynb/01_delta_method.ipynb)
  - Given the distribution of $X \sim D(\mu, \sigma^2)$, derive the variance of $g(X)$.
  - **Key words**: delta method
- [Bayes Estimate](https://github.com/houzhj/Statistics/blob/main/ipynb/01_bayes_estimate.ipynb)
  - Bayes estimates with different prior distributions, examples for Bernoulli and Possion distribution.
  - Compared Bayes estimates with Frequentist estimates (maximum likelihood estimate).
  - **Key words**: Bayes estimator, prior distribution, posterior distribution 
## 2. Hypothesis testing
- [Wald Test and Likelihood Ratio Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_wald_lr_test.ipynb)
  - Wald test and likelihood ratio test (definitions, derivations, codes, etc.) 
  - Examples with four distributions: Bernoulli, Binomial, Poisson, and Uniform.
  - **Key words**: Wald test, likelihood ratio test, sample size, Type 1 Error, Type 2 Error
- [Hypothesis Testing Example with a Uniform Distribution](https://github.com/houzhj/Statistics/blob/main/ipynb/02_test_with_uniform_distribution.ipynb)
  - As discussed above, the two maximum-likelihood-based tests can not be directly applied for uniform distribution. Investegate an alternative test.
  - Applications: (1) calculate the significant levels for given tests, and (2) find the rejection thresholds to achieve desired significant levels.
  - **Key words**: Uniform distribution, test design, properties of a test
- [Two Sample Mean Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_two_sample_mean_test.ipynb)
  - Conduct and compare several two-sample tests under different scenarios (equal/unequal sample size, equal/unequal variance)
  - **Key words**: two sample mean test
- [From Wald Test to  Student's t Test (T test)](https://github.com/houzhj/Statistics/blob/main/ipynb/02_wald_t_test.ipynb)
  - Failure of the Wald test with very small sample size. Using examples with Bernoulli and Normal distributions
  - Student's t test (definitions, derivations, codes, etc.)
  - **Key words**: Wald test, small sample size, Student's t test
- [Multiple Hypothesis Testing (i.e. simultaneously testing multiple hypotheses)](https://github.com/houzhj/Statistics/blob/main/ipynb/02_multiple_hypothesis_testing.ipynb)
  - Multiple hypothesis testing (definitions, derivations, codes, etc.)
  - Two commonly used methods in multiple hypothesis testing: Family Wise Error Rate (FWER) and False Discovery Rate (FDR).
  - **Key words**: multiple hypothesis testing, FWER, FDR, Bonferroni Correction, Benjamin–Hochberg Method
- [(Goodness of Fit Test) Pearson's Chi-squared Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_pearson_chi_squared_test.ipynb) 
  - Pearson's Chi-squared test (definitions, derivations, codes, etc.)
  - Examples with Categorial distribution and Poisson distribution
  - **Key words**: Pearson's chi-squared test, categorical distribution, Poisson distribution, degree of freedom
- [(Goodness of Fit Test) Kolmogorov-Smirnov Test and Lilliefors Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_ks_lilliefors_test.ipynb)
  - Kolmogorov-Smirnov test and Lilliefors test (definitions, derivations, codes, etc.)
  - Examples with Normal distribution and Uniform distribution
  - **Key words**: Kolmogorov-Smirnov test, Lilliefors test
## 3. Linear Regression
- Joint Distribution
- Linear Regression: Estimate and T Test

