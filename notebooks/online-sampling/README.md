# Online sampling

(aka active learning?)

Approaches?

- Model uncertainty and tradeoff exploitation versus exploration (something like ucb).
- RL to pick locations.

### Representing the uncertainty of an MRI

Uncertainty comes from two sources;
- noise,
- underconstrained.

All samples in an underconstrained set should have equal probability!?
Noise should induce a distribution which would diffuse out from the samples in an underconstrained set.

### RL

Action = the possible places to sample.


## Questions


- How much time do we have? How long does it take for the MRI to take a sample?
- Which dynamics of the MRI process matter here? Slew rates, ...?


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
