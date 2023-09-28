# Three Solutions to Calculate Confidence Intervals
### ipynb file can be found [here]()

Consider a hypothesis testing with a random variable $X \sim U[a,b]$, with 
$$H_0: b = c$$
$$H_1: b \leq c$$
where $c$ is a given constant. Also, let's assume $a$ is known, so in this case the test is mainly about $b$. 
Assume we have observations 
$$X_1,…,X_n$$ 
or 
$$X_{(1)} \leq X_{(2)} \leq ... \leq X_{(n)}$$
where $X_{(i)}$ is the $i^{th}$ largest number among $X_1,...,X_n$

Clearly, $a \leq X_{(1)} \leq X_{(n)} \leq c$. If any observed $X_i>c$, we can directly conclude that the null hypothesis is true. So let assume 
$$X_{(1)} \leq ... \leq X_{(n)} \leq c$$


## 1. Test Statistic
The maximum likelihood estimator of the upper bound parameters $b$ can be derived as below. 

The likelihood function
$$L(a,b) = \prod_{i=1}^{n} \dfrac{1}{b-a} = (b-a)^{-n}$$
$$\downarrow$$
The log-likelihood function
$$l(a,b) = -n \times ln(b-a)$$
$$\downarrow$$

$$\dfrac{\partial l(a,b)}{\partial b} = \dfrac{-n}{b-a}<0$$

So $L(a,b)$ is monotonically decreasing with respect to $b$, and $L(a,b)$ is maximized at the smallest possible value of $b$. 
According to observations $(X_1,...,X_n)$,
$$\hat{b}= max(X_1,...,X_n)=X_{(n)}$$

So any test we could do might involve the test statistic $\hat{b}=X_{(n)}$. Let's just use $X_((n))$ as the test statistic.

$$T_n = X_((n))$$ 

## 2. Rejection Region
The reject region $R$ is defined so that Reject $H_0$ if $T_n = X_{(n)} \in R$. Consider the rejection region defined by a scalar $r$, reject the null hypothesis $H_0: b=c$ if
$$T_n \in [a,c]$$

Note that we must have $a < r < c$, otherwise the test is logical meaningless. 
- If we chose $r < a$: it is impossible that the maximum of $X_1,...,X_n$ is smaller than a number that is smaller than the lower bound $a$. I.e., the null hypothesis will always be rejected. 
- If we chose $r > c$: the null hypothesis will be rejected if we observed $X_{(n)}$ that is greated than $r$, which is greater than $c$. In this case we can conclude that the null hypothesis is true withtout any statistical testing, and the test is essentially unnecessary.  

## 3.Type 1 and Type 2 Error
The probability of Type 1 and Type 2 Error of this tests can be expressed by functions of known variables ($a,r,c,n$). In specific
- $a$: the lower bound of the uniform distribution
- $r$: the scalar that defines the reject region, reject the $H_0$ if $T_n = X_{(n)} \in [a,r]$
- $c$: the value in the null hypothesis, $H_0: b=c$, $H_1: b < c$
- $n$: the number of observations in the sample

According to the [definition](https://en.wikipedia.org/wiki/Power_of_a_test) to Type 1 (and significant level) and Type 2 errors (and power), 

- The significant level of this test is the probability of rejecting $H_0$ ($T_{(n)} \in [a,r]$) when $H_0$ is true ($b \geq c$).

$$\alpha = P(a < T_n < r) | b \geq c) = P(a < X_{(n)} < r) | b \geq c) $$

$$ =  P[ (a < X_1 < r) | b \geq c) \cap (a < X_2 < r) | b \geq c) \cap ... \cap (a < X_n < r) | b \geq c)]$$ 

$$ = \left(\dfrac{r-a}{c-a} \right)^n$$

- The power of this test is the 1 minus probability of not rejecting $H_0$ ($T_{(n)} \notin [a,r]$) when $H_0$ is false ($b < c$).

$$ 1-\beta = 1-P(T_n \geq r | b < c) = 1-[1-P(a < T_n < r | b < c)] = P(a <  X_{(n)} < r | b < c)$$ 

$$ = P[ (a < X_1 < r) | b < c) \cap (a < X_2 < r) | b < c) \cap ... \cap (a < X_n < r) | b < c)]$$

$$\geq P[ (a < X_1 < r) | b = c) \cap (a < X_2 < r) | b = c) \cap ... \cap (a < X_n < r) | b = c)]$$ 

$$ =  \left(\dfrac{r-a}{c-a} \right)^n$$

## 4.Applications
We can use these conclusions to 
- calculate the significant level of a test, or
- design a test(i.e., finding the rejection region) to achieve a given significant level. 

Assume we have the following information:

(1) The random variable $X \sim U(0,b)$, where $b$ is the unknown upper bound, and is the parameter of interest.

(2) We have $n=10$ observations $X_1, ..., X_{10}$. The sample size can be any positive integer, and we fix it at 10. 

(3) We observe the maximum value among $X_1, ..., X_{10}$. Then we have the hypotheses $H_0: b = c$, $H_1: b < c$. Note that we will only use $c > max(X_1,...,X_{10})$ in the hypotheses, otherwise the test will be meaningless - no one needs to test if $b=4$ after observing the sample maximum at 5. Instead, one might be interested in whether the upper bound parameter is 6.  

### Calculating the significant level of a given test 
Let's say we have a testing with pre-specified rejection region $[5,9]$, and hypotheses $H_0: b=10$, $H_1: b < 10$,. 

If the maximum in the sample $(n=8)$ is greater then 10, we can direcly conclude that $b>10$ and there is no need to conduct this test. 

Otherwise, we will reject $H_0: b=10$ if $X_{(10)}<9$. 

In this test, we have $a=5$, $n=8$, $c=10$, $r=9$, so the significant level is 
$$\left(\dfrac{r-a}{c-a} \right)^n = \left(\dfrac{9-5}{10-5} \right)^8 = 0.8^8 = 0.167772$$

The following codes calculate the significant level by simulation experiments. Since the significant level is for Type 1 error, the actual value of the parameter is the same as the value in $H_0$.

```python
a      = 5
r_list = [9+0.1*i for i in range(10)]
sample_size_list = [10,20,50,100]
h0     = 10

```

















