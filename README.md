# Statistics
These studies are about various topics in statistics, including mathematical derivations of key statistical conclusions and their applications, implementation using Python code, and case studies based on simulations.

## 1. Some important concepts in statistics
- Central Limit Theorem
- Distribution
- [Estimations and Confidence Interval](https://github.com/houzhj/Statistics/blob/main/ipynb/01_confidence_intervals.ipynb)
  - Three methods to calculate the confidence interval of a parameter: Conservative, Solve, Plug-in.
  - Using simulation to conduct and compare the three methods.
- [Delta Method](https://github.com/houzhj/Statistics/blob/main/ipynb/01_delta_method.ipynb)
  - Calculate the variance of a random variable (which is a function of another random variables with known distribution).
  - Compare the theoretical variance (based on the Delta method) with the simulation-based variances.
- Bayes Estimate

## 2. Hypothesis testing
- [Wald Test and Likelihood Ratio Test](https://github.com/houzhj/Statistics/blob/main/ipynb/02_wald_lr_test.ipynb)
  - Compare the two maximum-likelihood-based testing methods using simulated testing data, in terms type 1 and type 2 error rates.
  - Investigate the impact of sample sizes.
  - Consider the hypothesis testing about the parameters of four distributions: Bernoulli, Binomial, Poisson, and Uniform.
  - Demonstrated that the test statistic cannot be applied to case with Uniform distribution . 
- [Hypothesis Testing Example with a Uniform Distribution](https://github.com/houzhj/Statistics/blob/main/02_Testing_with_Uniform.md)
  - As discussed above, the Wald Test and the maximum-likelihood-based tests can not be directly applied for uniform distribution.
  - Investegate alternative testing methods.
  - In this testing context, calculate the significant levels for given tests, and find the reject thresholds to achieve desired significant levels. 
- [Two Sample Mean Test](https://github.com/houzhj/Statistics/blob/main/02_Two_Sample_Mean_Tests.md)
  - Conduct and compare several two-sample tests under different scenarios (equal/unequal sample size, equal/unequal variance)
- From Wald Test to  Student's t test (T test)
- Multiple Hypothesis Testing (i.e. simultaneously testing multiple hypotheses)
- Goodness of Fitness Tests

## 3. Linear Regression
- Joint Distribution
- Linear Regression: Estimate and T Test

