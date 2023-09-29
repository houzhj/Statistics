# Delta Method

### The contents of this note
- **Derive the variance of a random variable using Delta Method**
- **Compare the theoretical variance (based on the Delta Method) with the simulation-based variances**

$$$$

$$$$

Let $X_1,...X_n$ denote a random sample of independent observations from a population with mean $\mu$ and variance $\sigma^2$. 
According to Central Limit Theorem (CLT), as $n \rightarrow \infty$
$$\sqrt{n}(\overline{X}-\mu) \rightarrow N(0,\sigma^2)$$

Let $g()$ be a function that is continuously differentiable at $\mu$, and its first derivative evaluated at $\mu$ is non-zero, i.e, $g'(\mu)$ exists and $g'(\mu) \ne 0$.

Then, according to the Delta Method,

$$\sqrt{n}(g(\overline{X})-g(\mu)) \rightarrow N(0,\sigma^2 g'(\mu)^2)$$

## Part 1 - Derive the variance of a random variable using Delta Method
**Two distribution (of the original random variable) are consider:**
- Normal
- Exponential

**Three $g(.)$ functions applied to the original random variable to generate a new random variable) are considered:**
- $g(x) = 2x$
- $g(x) = 1/x$
- $g(x) = x^2$

All of these funcstion satisfies that $g'(\mu)$ exists and $g'(\mu) \ne 0$.

### Example 1 - Normal Distribution (10,5)
$$\sqrt{n}(\overline{X}-\mu) \rightarrow N(0,\sigma^2)$$

**Function 1:**

$g(x) = 2x \rightarrow g'(x)=2 \rightarrow g'(x)^2=4 \rightarrow g'(\mu)^2=4$

$\downarrow$

$\sqrt{n}(g(\overline{X})-g(\mu)) = \sqrt{n}(2\overline{X}-2\mu)\rightarrow N(0,\sigma^2 \times 4=N(0,5 \times 4))$


**Function 2:**

$g(x) = \frac{1}{x} \rightarrow g'(x)=-x^{-2} \rightarrow g'(x)^2=x^{-4} \rightarrow g'(\mu)^2=\mu^{-4}$

$\downarrow$

$\sqrt{n}(g(\overline{X})-g(\mu)) = \sqrt{n}(\frac{1}{\overline{X}}-\frac{1}{\mu})\rightarrow N(0,\sigma^2 \times \mu^{-4}=N(0,5 \times 10^{-4}))$

**Function 3:**

$g(x) = x^2 \rightarrow g'(x)=2x \rightarrow g'(x)^2=4x^2 \rightarrow g'(\mu)^2=4\mu^2$

$\downarrow$

$\sqrt{n}(g(\overline{X})-g(\mu)) = \sqrt{n}((\overline{X})^2-\mu^2)\rightarrow N(0,\sigma^2 \times 4\mu^2=N(0,5 \times 4 \times 10^2))$

### Example 2 - Exponential Distribution (5)
$$\sqrt{n}\left(\overline{X}-\dfrac{1}{\lambda}\right) \rightarrow N\left(0,\dfrac{1}{\lambda^2}\right)$$

**Function 1:**

$g(x) = 2x \rightarrow g'(x)=2 \rightarrow g'(x)^2=4 \rightarrow g'(\dfrac{1}{\lambda})^2=4$

$\downarrow$

$\sqrt{n}(g(\overline{X})-g(\dfrac{1}{\lambda})) = \sqrt{n}(2\overline{X}-\dfrac{2}{\lambda})\rightarrow N(0,\dfrac{1}{\lambda^2} \times 4)=N(0,\dfrac{4}{5^2})$


**Function 2:**

$g(x) = \frac{1}{x} \rightarrow g'(x)=-x^{-2} \rightarrow g'(x)^2=x^{-4} \rightarrow  g'(\dfrac{1}{\lambda})^2=\lambda^4$

$\downarrow$

$\sqrt{n}(g(\overline{X})-g(\dfrac{1}{\lambda})) = \sqrt{n}(\frac{1}{\overline{X}}-\frac{1}{1/\lambda})\rightarrow N(0,\dfrac{1}{\lambda^2} \times \lambda^4) =N(0,\lambda^2)=N(0,5^2)$

**Function 3:**

$g(x) = x^2 \rightarrow g'(x)=2x \rightarrow g'(x)^2=4x^2 \rightarrow  g'(\dfrac{1}{\lambda})^2=\dfrac{4}{\lambda^2}$

$\downarrow$

$\sqrt{n}\left(g(\overline{X})-g(\dfrac{1}{\lambda})\right) = \sqrt{n}\left((\overline{X})^2-\left(\dfrac{1}{\lambda}\right)^2\right) \rightarrow N\left(0,\frac{1}{\lambda^2} \times \dfrac{4}{\lambda^2} \right)=N\left(0,\dfrac{4}{\lambda^4}\right)$


## Part 2 - Compare the theoretical variance (based on the Delta Method) with the simulation-based variances
This analysis is conducted in Python. The codes can be found [here]():

The codes work like below. We will do 1000 experiments. In each experiment, first generate a sample with 1000 obserbatins for the original random variable and calculate the sample mean. 

```python

n_experiment = 1000
sample_size  = 1000

mu           = 10
sigma2       = 5
population   = np.random.normal(mu,np.sqrt(sigma2),100000)
sample_mean  = [np.nan]*n_experiment
for i in range(n_experiment):
    sample_now     = np.random.choice(population, size=sample_size,replace=True)
    sample_mean[i] = np.mean(sample_now)
```

Then calculate the values of $\sqrt{n}(g(\overline{X})-g(\mu))$ for each $g(.)$

```python
Theoretical_mean = mu
series_1 = [np.sqrt(sample_size)*(sample_mean[i]-Theoretical_mean) for i in range(n_experiment)]
series_2 = [np.sqrt(sample_size)*(2*sample_mean[i]-2*Theoretical_mean) for i in range(n_experiment)]
series_3 = [np.sqrt(sample_size)*(1/sample_mean[i]-1/Theoretical_mean) for i in range(n_experiment)]
series_4 = [np.sqrt(sample_size)*(sample_mean[i]**2-Theoretical_mean**2) for i in range(n_experiment)]
```

So we have 1000 simulated values of $t=\sqrt{n}(g(\overline{X})-g(\mu))$, then 
- create a histogram of $t$
- compare the theoretical variance (based on the Delta Method) and the simulation-based variance of $t$.

```python
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_2,alpha=0.5,bins=30,label = 'g(x)=2x')
plt.title('Histogram of t, where g(x) = 2x')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(4*sigma2))
print("Sample Variance: " + str(round(np.var(series_2,ddof=1),3)))
```




