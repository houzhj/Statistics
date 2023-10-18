# Statistics
These studies are about various topics in statistics, including mathematical derivations of key statistical conclusions and their applications, implementation using Python code, and case studies based on simulations.

## 1. Some important / interesting (or both) concepts in statistics
- Central Limit Theorem
- Distribution
- [Estimations and Confidence Interval](https://github.com/houzhj/Statistics/blob/main/ipynb/01_confidence_intervals.ipynb)
  - Three methods to calculate the confidence interval of a parameter: Conservative**, Solve, Plug-in.
  - **Key words**: Confidence interval
- [Delta Method](https://github.com/houzhj/Statistics/blob/main/ipynb/01_delta_method.ipynb)
  - Given the distribution of $X \sim D(\mu, \sigma^2)$, derive the variance of $g(X)$.
  - **Key words**: Delta method
- [Bayes Estimate](https://github.com/houzhj/Statistics/blob/main/ipynb/01_bayes_estimate.ipynb)
  - Bayes estimates with different prior distributions, examples for Bernoulli and Possion distribution.
  - Compared Bayes estimates with Frequentist estimates (maximum likelihood estimate).
  - **Key words**: Bayes estimator, Prior distribution Posterior distribution
- [Estimation with latent variables](https://github.com/houzhj/Statistics/blob/main/ipynb/01_estimation_with_latent_variables.ipynb)
  - Examples of estimating a parameter using observations of a latent variable
  - Given that $X_i\sim Exp(\lambda)$ where $\lambda$ is an unknown parameter to estimate. Use the observations of a latent variable $Y_i$, instead of observations of $X_i$.
  - **Key words**: Latent variables, Exponential distribution
## 2. Hypothesis testing
- [Wald Test and Likelihood Ratio Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_wald_lr_test.ipynb)
  - Wald test and likelihood ratio test (definitions, derivations, codes, etc.) 
  - Examples with four distributions: Bernoulli, Binomial, Poisson, and Uniform.
  - **Key words**: Wald test, Likelihood ratio test, Sample size, Type 1 Error, Type 2 Error
- [Hypothesis Testing Example with a Uniform Distribution](https://github.com/houzhj/Statistics/blob/main/ipynb/02_test_with_uniform_distribution.ipynb)
  - As discussed above, the two maximum-likelihood-based tests can not be directly applied for uniform distribution. Investegate an alternative test.
  - Applications: (1) calculate the significant levels for given tests, and (2) find the rejection thresholds to achieve desired significant levels.
  - **Key words**: Uniform distribution, Test design, Properties of a test
- [Two Sample Mean Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_two_sample_mean_test.ipynb)
  - Conduct and compare several two-sample tests under different scenarios (equal/unequal sample size, equal/unequal variance)
  - **Key words**: Two sample mean test
- [From Wald Test to  Student's t Test (T test)](https://github.com/houzhj/Statistics/blob/main/ipynb/02_wald_t_test.ipynb)
  - Failure of the Wald test with very small sample size. Using examples with Bernoulli and Normal distributions
  - Student's t test (definitions, derivations, codes, etc.)
  - **Key words**: Wald test, Small sample size, Student's t test
- [Multiple Hypothesis Testing (i.e. simultaneously testing multiple hypotheses)](https://github.com/houzhj/Statistics/blob/main/ipynb/02_multiple_hypothesis_testing.ipynb)
  - Multiple hypothesis testing (definitions, derivations, codes, etc.)
  - Two commonly used methods in multiple hypothesis testing: Family Wise Error Rate (FWER) and False Discovery Rate (FDR).
  - **Key words**: Multiple hypothesis testing, FWER, FDR, Bonferroni Correction, Benjamin–Hochberg Method
- [(Goodness of Fit Test) Pearson's Chi-squared Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_pearson_chi_squared_test.ipynb) 
  - Pearson's Chi-squared test (definitions, derivations, codes, etc.)
  - Examples with Categorial distribution and Poisson distribution
  - **Key words**: Pearson's chi-squared test, Categorical distribution, Poisson distribution, Degree of freedom
- [(Goodness of Fit Test) Kolmogorov-Smirnov Test and Lilliefors Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_ks_lilliefors_test.ipynb)
  - Kolmogorov-Smirnov test and Lilliefors test (definitions, derivations, codes, etc.)
  - Examples with Normal distribution and Uniform distribution
  - **Key words**: Kolmogorov-Smirnov test, Lilliefors test
## 3. Regression
- [Joint Distribution](https://github.com/houzhj/Statistics/blob/main/ipynb/03_joint_distribution.ipynb)
  - Concepts in the context of joint distribution (bivariate)
  - Creating random samples from given probability density function (univariate pdf or joint pdf)
  - **Key words**: Joint distribution, Random Variable generation
- [Linear Regression](https://github.com/houzhj/Statistics/blob/main/ipynb/03_linear_regression.ipynb)
  - Least Square Estimator (distribution, expression, derivations, codes, etc.)
  - Hypothesis testing in linear regressions (definitions, derivations, codes, etc.). Regular hypothesis (e.g., $\beta_1=1$), two-coefficients hypothesis (e.g, $\beta_1=\beta_2$), and multiple testing ($0<\beta_1<\beta_2$).
  - **Key words**: Least Square Estimator, T test for linear regression models
- [Generalized Linear Regression](https://github.com/houzhj/Statistics/blob/main/ipynb/03_generalized_linear_regression.ipynb)
  - Mathematics behind generalized linear regression (definitions, derivations, etc.)
  - **Key words**: Exponential family, Link function and canonical link function

## 4. Interests and hobbies
- [Texas hold 'em](https://github.com/houzhj/Statistics/blob/main/ipynb/IH_TexasHoldem.ipynb)
  - Python codes for basic poker game actions, such as simulating a two-player game, generating random hands, comparing hand values, etc.
  - Design and test a strategy for simplified games (two player; no betting; order of action is not considered).
  - Estimate the probabilities of hands (by simulation)
  - Estimate the winning rate for a starting hands (by simulation)



    
