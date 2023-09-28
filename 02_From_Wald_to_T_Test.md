# From Wald Test to Student's t test (T test)
### ipynb file can be found [here]()

# 1. The Wald Test
Wald test is a maximum likelihood estimate based test. It is based on the asymptotic normal approximation to the maximum likelihood estimator.
Two equivalent expression of the Wald test statistics are shown below
Consider the parameter $\theta$. The maximum likelihood of $\theta$ is $\hat{\theta}$
The hypotheses are given below (the hypothesis can be one-sided too)
$$H_0: \theta = \theta_0$$
$$H_1: \theta \ne \theta_0$$

(1) $W = \sqrt{n} \times \sqrt{I(\theta)}(\hat{\theta}-\theta) \sim N(0,1)$ or $W = n \times I(\theta)(\hat{\theta}-\theta) \sim \chi^2(1)$

where $I(\theta)$ is the Fisher Information of $\theta$. Normally we can plug in $I(\hat{\theta})$ into the expression. 
 
(2) $W =  \dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})}\sim  N(0,1)$ or $W = \dfrac{(\hat{\theta}-\theta_0)^2}{var(\hat{\theta})} \sim \chi^2(1)$



It can be shown that these expressions are equivalent. 
That asymptotic normal approximation for the MLE says
$$\sqrt{n}(\hat{\theta}-\theta) \sim N(0,I^{-1}(\theta))$$
$$\downarrow$$
$$var(\sqrt{n} \times \hat{\theta}) = I^{-1}(\theta)$$
$$\downarrow$$
$$var(\hat{\theta}) = \frac{1}{\sqrt{n}} \times I^{-1}(\theta) = \dfrac{1}{n I(\theta)}$$
Plugging this to the first expression yields
$$W = \sqrt{n} \times \sqrt{I(\theta)}(\hat{\theta}-\theta) = \sqrt{n I(\theta)}  \times (\hat{\theta}-\theta) =\sqrt{\dfrac{1}{var(\hat{\theta})}} \times (\hat{\theta}-\theta) = \dfrac{(\hat{\theta}-\theta_0)}{se(\hat{\theta})} $$

Consider two examples below. These examples show that a Wald test does not work well when the sample size is very small. We consider hypothesis testing about the parameters from two distributions. 
- Bernoulli distribution 
- Normal distribution (Student's t test is discussed in this example)
  
This study is based on both analytical derivations and simulations.












