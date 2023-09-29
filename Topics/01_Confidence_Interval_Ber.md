## Example A - Bernoulli Distribution
$$X \sim Ber(p)$$
The maximum likelihood of $p$ is
$$\hat{p} = \overline{X}$$

By central limit theorem, $\sqrt{n}(\overline{X}-p) \rightarrow N(0,p(1-p))$. 

The confidence interval of $p$ can be written as 
$$\overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$


### 1. **Conservative Bound**:
   
   Since $\sqrt{p(1-p)}\leq\sqrt{0.5(1-0.5)}=0.5$ when $p \in (0,1)$, we have
   $$CI_{cons} = \overline{X} \pm q_{1-\alpha/2}\dfrac{0.5}{\sqrt{n}}$$
   
### 2. **Solve**

   According to the following derivations
   $$\overline{X} - q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}} \leq  p \leq \overline{X} + q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$
   $$\downarrow$$
   $$- q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}} \leq  p-\overline{X} \leq  + q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}$$
   $$\downarrow$$
   $$(p-\overline{X})^2 \leq  \left( q_{1-\alpha/2} \dfrac{\sqrt{p(1-p)}}{\sqrt{n}}\right)^2$$
   $$\downarrow$$
   $$Ap^2+Bp+c \leq 0$$
   where
   $$A=1+\dfrac{(q_{1-\alpha/2})^2}{n}, B=-2\overline{X}-\dfrac{(q_{1-\alpha/2})^2}{n}, C=(\overline{X})^2$$
   $$\downarrow$$

   The confidence interval is 
   $$CI_{solve} = \left( \dfrac{-B \pm \sqrt{B^2-4AC}}{2A} \right)$$

### 3. **Plug-in**
   Given that $\hat{p}=\overline{X}$, the confidence interval is
   $$CI_{plug-in} = \overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{n}} =\overline{X} \pm q_{1-\alpha/2} \dfrac{\sqrt{\overline{X}(1-\overline{X})}}{\sqrt{n}}$$

