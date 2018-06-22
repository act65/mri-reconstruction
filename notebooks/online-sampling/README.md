# Online sampling

(aka active learning?)

Approaches?

- Model uncertainty and tradeoff exploitation versus exploration (something like ucb).
- RL to pick locations.
- Differentiable sampling?

## Representing the uncertainty

Uncertainty comes from two sources;
- noise,
- underconstrained.

All samples in an underconstrained set should have equal probability!?
Noise should induce a distribution which would diffuse out from the samples in an underconstrained set.


## Online sampling

(Learn to) iteratively pick the best points to samples that minimise uncertainty.  

$$
\begin{align}
s_t  &= f(\theta, \tilde x_t, ?) \tag{pick next sample}\\
\tilde x_t &= g(s_0, \dots, s_t) \tag{reconstruct}\\
l_t &= d(x, \tilde x_t) + H(p(\tilde x_t)) \tag{error and uncertainty}\\
\text{regret} &= \sum_0^T l_t(?)- \mathop{\text{min}}_{s}\sum_0^T l_t(?)\\
\theta^* &= \mathop{\text{argmin}}_{\theta} \text{regret}
\end{align}
$$

We can search for the sample that best reduces error and uncertainty.

#### Learning the basis
What about learning the right basis to use to sample!?
- the magnetic resonance pertubations.


### Theory

__Want to show__

The lower bound on error (wrt samples) in the online setting < lower bound with random samples.

How uch better can you do if you get to pick the samples?
- all in one go,
- iteratively.


## MRI specific


- How much time do we have? How long does it take for the MRI to take a sample?
- Which dynamics of the MRI process matter here? Slew rates, ...?
