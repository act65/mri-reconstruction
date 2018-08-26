Why is this hard?
TODO pics from notebook

#### Normalisation

The problem is normalising. Want a fn/representation that can easily normalise the data. If we increase the probability of a location, that should decrease the probability of other locations.

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


...

$$
\begin{align}
p(f(x)) = p(x) \cdot \mid \frac{df}{dx} \mid \\
\end{align}
$$
But why? Simple example. $f(x) = 2x$. Then we suddenly have ...? (DL book!?)

You could call

## Normalising flows

But want to decompse the prior.
Masked autoregressive.
