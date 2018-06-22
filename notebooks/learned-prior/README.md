## Inverse learning

Given a dataset $\{x_i, y_i: x_i \in X, y_i \in Y\}$ and the forward process $f: X \to Y$ construct an approximation to $f^{-1}: Y \to X$.

Let $X= \mathbb R^{n}$ and $Y=\mathbb R^{m}$ then set $n >> m$. Want to minimize $m$ while maintaining the ability to accurately reconstruct $x_i$.

<!-- So we have control over the forward function...!? -->

Similar to compressed sensing, but we also have access to;
- the $x_i$s
- $df$.


Compressed sensing

$$

$$

Solutions
- learned prior - GAN
- learned sparse basis - AE



## Learning a better prior

__Q__ How to incorporate the priors!?!? Which approach is better? What are the advantages/disadvantages?

- Meta learning is used to learn how to correct the imperfect $P(x \mid y)$
- GANs to learn $P(x), P(y)$  
- Distill them info a learned forward model $NN(y)$


### Meta

Meta learning needs to learn to correct $\frac{dL}{dx}$


### GANS

But how do we learn $P(X)$? In reality we dont really have the ground truth $x$ values, only noisy samples/reconstructions from $y$.

GANs add an extra loss, maybe that provides some useful info!?


### Derivation of learned prior for CS

$$
\begin{align}
P(x \mid y) &= \frac{P(y \mid x )P(x)}{P(y)} \\
log P(x \mid y) &= logP(y \mid x ) + logP(x) - logP(y) \\
x^* &= \mathop{\text{argmax}}_x log P(x \mid y)\\
\\
-logP(y \mid x) &= e^{\parallel y - f(x) \parallel} \tag{or f(x) - y?} \\
-logP(x) &= e^{\parallel x\parallel} \\
-log P(x \mid y) &= \parallel y - f(x) \parallel + \parallel x \parallel \\
x^* &= \mathop{\text{argmin}}_x  \parallel y - f(x) \parallel + \parallel x \parallel\\
\end{align}
$$

__TODO__ not convinced you can throw away $logP(y)$
