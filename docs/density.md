Why is density estimation hard?
In high dimensions are the many directions and a large amount of volume.

## Maximum likelihood

In an ideal world we would like the ability to approximate arbitrary distributions. To learn $f: X \to Y$ such that the probability density is preserved. An easy and intuitive way to learn a density model is by maximum likelihood estimation.

$$
\hat \theta = \mathop{\text{argmax}}_{\theta} \mathbb E_{x\sim D} \left[ p_{\theta}(x)\right]
$$

Thus, models that allocates higher probability to observed $x$s are better. However, the class of functions that $\theta$ can represent is limited. We would like to use an abitrary functon approximator like neural networks, but they fail at ML because they can simply predict $nn(x_i) = \infty$ for all inputs. They can do this because they are not normalised.

A simple representation like a tensor that is indexed by possible $x$s has a similar normalisation problem (naively optimising it to do ML will give $\infty$). But, it can be easily constrained/regualised to give normalised results.

$$
p(x) = T[x] \\
\hat T = \mathop{\text{argmax}}_{T} \mathbb E_{x\sim D} \left[ p_{\theta}(x)\right] \text{s.t.} \sum_{i, \dots} T_{i, \dots} = 1
$$

Which might be implemented as simply the decay of each element towards zero probabilty. But, just for mnist $T$ would need have $(28 \times 28)^{256}$ elements for each possible image (which according to Google's calculator is infinity...).


In principle this idea could be applied to neural networks as well.

So, if we increase the probability of a location, that should decrease the probability of other locations.

$$
\begin{align}
p(x) = \frac{f_{\theta}(x)}{\int f_{\theta}(x) dx} \\
\end{align}
$$

Empirical estimates of $f$ when it is a NN. Hmm, not sure how to do that...

Want to find a parameterised fn that is easily integrated. Oh, how about $e^{x}$...

$$
\begin{align}
f(x) &= g_n( \dots g_1(g_0(x))) \\
\int f(x) dx &= ?? \\
\end{align}
$$

Want some sort of decomposition of the integral into something nicer.
Want an analytical way to calculate it!?

## Normalising flows


$$
p(f(x)) = \frac{1}{det(J(f))}p(x) \\
$$


But want to decompse the prior.
Masked autoregressive.

Problems.
- Too much non linearity makes ...
- Must be ivertible!?
