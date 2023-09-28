# Delta Method
### ipynb file can be found [here](https://github.com/houzhj/Statistics/blob/main/ipynb/01_delta_method.ipynb)

### A brief introduction to the Delta Method (Univariate)
https://en.wikipedia.org/wiki/Delta_method

Let $X_1,...X_n$ denote a random sample of independent observations from a population with mean $\mu$ and variance $\sigma^2$. 
According to Central Limit Theorem (CLT), as $n \rightarrow \infty$
$$\sqrt{n}(\overline{X}-\mu) \rightarrow N(0,\sigma^2)$$

Let $g()$ be a function that is continuously differentiable at $\mu$, and its first derivative evaluated at $\mu$ is non-zero, i.e, $g'(\mu)$ exists and $g'(\mu) \ne 0$.

Then, according to the Delta Method,

$$\sqrt{n}(g(\overline{X})-g(\mu)) \rightarrow N(0,\sigma^2 g'(\mu)^2)$$

### The goals of this study are to 
- Calculate the variance of a random variable (which is a function of another random variables with known distribution, including the true values of the distribution parameters).
- Compare the theoretical variance (based on the Delta Method) with the simulation-based variances.

### Two distribution (of the original random variable) are consider:
- Normal
- Exponential

### Three $g(.)$ functions applied to the original random variable to generate a new random variable) are considered:
- $g(x) = 2x$
- $g(x) = 1/x$
- $g(x) = x^2$

All of these funcstion satisfies that $g'(\mu)$ exists and $g'(\mu) \ne 0$.

## Examples

```python
n_experiment = 1000
sample_size  = 1000
```

## Example 1 - Normal Distribution (10,5)
$$\sqrt{n}(\overline{X}-\mu) \rightarrow N(0,\sigma^2)$$

Generate a sample with 1000 obserbatins for the original random variable and calculate the sample mean. 

Do this experiment 1000 times. 
```python
mu     = 10
sigma2 = 5
population  = np.random.normal(mu,np.sqrt(sigma2),100000)
sample_mean = [np.nan]*n_experiment
for i in range(n_experiment):
    sample_now     = np.random.choice(population, size=sample_size,replace=True)
    sample_mean[i] = np.mean(sample_now)
```

For each experiment, calculate the values of $\sqrt{n}(g(\overline{X})-g(\mu))$ for each $g(.)$
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

### Function 1

$$g(x) = 2x \rightarrow g'(x)=2 \rightarrow g'(x)^2=4 \rightarrow g'(\mu)^2=4$$ 

$$\downarrow$$

$$\sqrt{n}(g(\overline{X})-g(\mu)) = \sqrt{n}(2\overline{X}-2\mu)\rightarrow N(0,\sigma^2 \times 4=N(0,5 \times 4))$$

```python
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_2,alpha=0.5,bins=30,label = 'g(x)=2x')
plt.title('Histogram of t, where g(x) = 2x')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(4*sigma2))
print("Sample Variance: " + str(round(np.var(series_2,ddof=1),3)))
```

<img width="371" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/eabbc803-e42c-4793-b8a3-92cd7930d92a">


### Function 2


$$g(x) = \frac{1}{x} \rightarrow g'(x)=-x^{-2} \rightarrow g'(x)^2=x^{-4} \rightarrow g'(\mu)^2=\mu^{-4}$$ 

$$\sqrt{n}(g(\overline{X})-g(\mu)) = \sqrt{n}(\frac{1}{\overline{X}}-\frac{1}{\mu})\rightarrow N(0,\sigma^2 \times \mu^{-4}=N(0,5 \times 10^{-4}))$$

```python
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_3,alpha=0.5,bins=30,label = 'g(x)=1/x')
plt.title('Histogram of t, where g(x) = 1/x')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(sigma2*mu**(-4)))
print("Sample Variance: " + str(round(np.var(series_3,ddof=1),5)))
```

<img width="367" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/852a378e-f0f5-4dd1-9969-02ec08bc3ea3">


### Function 3

$$g(x) = x^2 \rightarrow g'(x)=2x \rightarrow g'(x)^2=4x^2 \rightarrow g'(\mu)^2=4\mu^2$$ 

$$\sqrt{n}(g(\overline{X})-g(\mu)) = \sqrt{n}((\overline{X})^2-\mu^2)\rightarrow N(0,\sigma^2 \times 4\mu^2=N(0,5 \times 4 \times 10^2))$$

```python
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_4,alpha=0.5,bins=30,label = 'g(x)=x^2')
plt.title('Histogram of t, where g(x) = x^2')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(sigma2*4*mu**(2)))
print("Sample Variance: " + str(round(np.var(series_4,ddof=1),5)))
```
<img width="373" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/6332681f-1591-46a6-9346-5e002f0f1672">


## Example 2 - Exponential Distribution (5)
$$\sqrt{n}\left(\overline{X}-\dfrac{1}{\lambda}\right) \rightarrow N\left(0,\dfrac{1}{\lambda^2}\right)$$
Generate a sample with 1000 obserbatins for the original random variable and calculate the sample mean. 
```python
Lambda = 5

##### In np.random.exponential, to create a sample of Exp(Lambda), using scale = 1/Lambda
population = np.random.exponential(scale = 1/Lambda,size= 100000)

sample_mean = [np.nan]*n_experiment
for i in range(n_experiment):
    sample_now = np.random.choice(population, size=sample_size,replace=True)
    sample_mean[i] = np.mean(sample_now)
```

For each experiment, calculate the values of $\sqrt{n}(g(\overline{X})-g(\mu))$ for each $g(.)$

```python
Theoretical_mean = 1/Lambda
series_1 = [np.sqrt(sample_size)*(sample_mean[i]-Theoretical_mean) for i in range(n_experiment)]
series_2 = [np.sqrt(sample_size)*(2*sample_mean[i]-2*Theoretical_mean) for i in range(n_experiment)]
series_3 = [np.sqrt(sample_size)*(1/sample_mean[i]-1/Theoretical_mean) for i in range(n_experiment)]
series_4 = [np.sqrt(sample_size)*(sample_mean[i]**2-Theoretical_mean**2) for i in range(n_experiment)]
```


### Function 1

$$g(x) = 2x \rightarrow g'(x)=2 \rightarrow g'(x)^2=4 \rightarrow g'(\dfrac{1}{\lambda})^2=4$$ 

$$\downarrow$$

$$\sqrt{n}(g(\overline{X})-g(\dfrac{1}{\lambda})) = \sqrt{n}(2\overline{X}-\dfrac{2}{\lambda})\rightarrow N(0,\dfrac{1}{\lambda^2} \times 4)=N(0,\dfrac{4}{5^2})$$

```python
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_2,alpha=0.5,bins=30,label = 'g(x)=2x')
plt.title('Histogram of t, where g(x) = 2x')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(round((4*(1/Lambda)**2),3)))
print("Sample Variance: " + str(round(np.var(series_2,ddof=1),3)))
```
<img width="365" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/ad146ab5-3ffa-4c54-8fdf-abedd9dc905f">


### Function 2

$$g(x) = \frac{1}{x} \rightarrow g'(x)=-x^{-2} \rightarrow g'(x)^2=x^{-4} \rightarrow  g'(\dfrac{1}{\lambda})^2=\lambda^4$$ 

$$\sqrt{n}(g(\overline{X})-g(\dfrac{1}{\lambda})) = \sqrt{n}(\frac{1}{\overline{X}}-\frac{1}{1/\lambda})\rightarrow N(0,\dfrac{1}{\lambda^2} \times \lambda^4) =N(0,\lambda^2)=N(0,5^2)$$

```python
image-3.png
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_3,alpha=0.5,bins=30,label = 'g(x)=1/x')
plt.title('Histogram of t, where g(x) = 1/x')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(Lambda**2))
print("Sample Variance: " + str(round(np.var(series_3,ddof=1),3)))
```

<img width="363" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/3e252b15-b322-415a-b80b-79e90753fe56">



### Function 3

$$g(x) = x^2 \rightarrow g'(x)=2x \rightarrow g'(x)^2=4x^2 \rightarrow  g'(\dfrac{1}{\lambda})^2=\dfrac{4}{\lambda^2}$$ 

$$\sqrt{n}\left(g(\overline{X})-g(\dfrac{1}{\lambda})\right) = \sqrt{n}\left((\overline{X})^2-\left(\dfrac{1}{\lambda}\right)^2\right) \rightarrow N\left(0,\frac{1}{\lambda^2} \times \dfrac{4}{\lambda^2} \right)=N\left(0,\dfrac{4}{\lambda^4}\right)$$


```python
plt.hist(series_1,alpha=0.5,bins=30,label = 'original')
plt.hist(series_4,alpha=0.5,bins=30,label = 'g(x)=x^2')
plt.title('Histogram of t, where g(x) = x^2')
plt.legend()
plt.show()

print("Theoretical Variance: " + str(4/(Lambda**4)))
print("Sample Variance: " + str(round(np.var(series_4,ddof=1),3)))
```

<img width="368" alt="image" src="https://github.com/houzhj/Statistics/assets/33500622/9ee4e78c-e2ac-4889-ad3a-da42498208a5">















