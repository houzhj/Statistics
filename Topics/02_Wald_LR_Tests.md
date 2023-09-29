# Delta Method

### The contents of this note
- **Introduction to Wald test and Likelihood Ratio test**
- **Example**
  - **Bernoulli distribution**
  - **Binomial distribution**
  - **Poisson distribution**
  - **Uniform distribution**

$$$$

$$$$

## Part 1 - Introduction to Wald test and Likelihood Ratio test

### 1. The Wald Test
The Wald statistic (for a single parameter $\theta$) takes the following form:
$$W = \dfrac{(\hat{\theta}-\theta_0)^2}{var(\hat{\theta})}$$
Under the null $H_0: \theta = \theta_0$ the test statistic $W \sim \chi^2(1)$

An alternative expression of the $W$ statistic is the square root of the one above, i.e.,
$$W = \dfrac{(\hat{\theta}-\theta_0)}{\sqrt{var(\hat{\theta})}}=\dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})}$$

Under the null $H_0: \theta = \theta_0$ the test statistic $W \sim N(0,1)$

Now focus on this second expression.

### 2. Likelihood Test
The likelihood ratio test basically looks at how much more likely is the alternative hypothesis when compared to the null hypothesis. To do this we look at the likelihood ratio (LR), consider the ratio

$$LR =\dfrac{sup_{\theta \in \Theta} L(X_1,...,X_n|\Theta)}{sup_{\theta \in \Theta_0} L(X_1,...,X_n|\Theta_0)} > c > 1$$
where 
- the $\sup$ notation refers to the [supremum](https://en.wikipedia.org/wiki/Infimum_and_supremum)
- $\Theta$ is the entire parameter space
- $\Theta_0$ is the parameter space specified in $H_0$, which is a subset of $\Theta$
- $c$ is a constant determined by the significance level of our test and that constant is greater than 1 (becasue $\Theta_0 \in \Theta$).

If this ratio suggests that the alternative hypothesis (numerator) is much more likely than the null hypothesis (denominator), then we want to reject the null hypothesis. This can be rewritten as the following test statistic (the reason for taking log and multiplying by 2 is because we will use the Wilks' theorem)
$$2 \times ln \left[\dfrac{sup_{\theta \in \Theta} L(X_1,...,X_n|\Theta)}{sup_{\theta \in \Theta_0} L(X_1,...,X_n|\Theta_0)}\right]$$

which gives rises to the test statistic of a LR Test (expressed as a difference between the log-likelihoods
)
$$T_n = 2\times [\mathcal{l}(\hat{\theta})-\mathcal{l}(\theta)]$$

According to [Wilks’ theorem](https://en.wikipedia.org/wiki/Wilks%27_theorem), if $H_0$ is true, $T_n$ will be asymptotically chi-squared distributed with degrees of freedom equal to the difference in dimensionality of $\Theta$ and $\Theta_0$


## Part 2 - Examples
Next, we will perform and compare the Wald tests and the Likelihood Ratio (LR) tests through simulations. 
- We will consider several examples of hypothesis tests regarding parameters from different distributions.
  - Bernoulli distribution
  - Binomial distribution
  - Poisson distribution
  - Uniform distribution
- In each example, we will use a true null hypothesis as well as several false null hypotheses, to conduct experiments to calculate the probabilities of **Type 1 Error** and **Type 2 Error** for both testing methods.
- Additionally, we will investigate the impact of sample size on the tests.
- The significant level for all the tests are 5%.
- All the tests are two-sided, i.e., $H_0: \theta=$ certain value. 
- **Remark: Note that in this analysis, when we perform data generation and calculate type 1 and type 2 errors, we know the true values of the parameters. This differs from conducting hypothesis tests in real-world scenarios.**

























