## Motivation

Conjecture:
> there is a fundamental difference between traditional compressed sensing regularisers (TV, L1) and learned regularisers.  

The ability to add semantically meaningful information.

To add arbitrarily large amounts of information. Or arbitrarily large pertubations.

All we can really ask is that the image is consistent with the measurements.

## Intro to CS

Goal
$$
\begin{align}
\psi: &\mathbb R^n \rightarrow \mathbb R^m  \tag{n >> m}\\
y &= \psi(x) \tag{$x \in \mathbb R^n$} \\
\psi: &\mathbb R^n \rightarrow \mathbb R^p \\
\end{align}
$$

#### Many solutions

The problem: because our observations are much smaller that the item we with to reconstruct, there are many possible solutions.

$$
\begin{align}
\mathcal S &= \{x_i:  \parallel \psi(x) - y \parallel_2 \le \epsilon,  \forall x_i \in \mathbb R^n\} \tag{4} \\
\end{align}
$$


#### Priors


$$
\begin{align}
\mathop{\text{argmin}}_x &\parallel \phi(x) \parallel_1 \text{subject to} \parallel \psi(x) - y \parallel_2 \le \epsilon \\
\end{align}
$$

$$
\begin{align}
\mathop{\text{argmin}}_x & \parallel \psi(x) - y \parallel_2  + \lambda \parallel \phi(x) \parallel_1 \tag{2}\\
\end{align}
$$

$$
\begin{align}
\mathop{\text{argmin}}_x & \parallel \psi(x) - y \parallel_2  + \lambda (1-p(x)) \tag{3}\\
\end{align}
$$

## Fantasised information

> learned priors can add objects with marcoscopic structure into the reconstructed image.

ok, that is easily testable!?
want some simple examples of a specific shape being added to a reconstruction.

Under which conditions are macroscopic fantaies added?
What is the definition of macroscopic? How can it be measured?
If we take two images, on reconstructed with a learned prior and another reconstructed with TV.

Note, it is also possible for L1 regularisation to result in large 'global' changes. E.g. if we use the fourier basis then sparisity implies ...?

Want to show that optimising (2) picks a solution from $\mathcal S$. But, optimising (3) doesnt not guarantee that the solution is consistent with the observations. In fact it can be arbitrarily far away.


Relationship to; mode collapse, adversarial examples, ???
__Q__ How can you verify that all modes have been captured?

## Guarantees

> If we are going to use learned priors I want some guarantees on what information can be added into the reconstruction.

What form could the guarantees take? Bounds, empirical tests(?), ?

#### Bounds

If we are detecting tumors in MRI scans we want bounds on the number of false positives/false negatives introduced by the number of samples used for reconstruction.

Learned prior type X: $n_{mistakes} = \mathcal O(\log{n_{samples}})$.

Have a dataset $X \in R^{n \times m}$.

$\mathcal A_{learned}$
$\mathcal A_{?}$


#### Empirical

Given a classifier and a labelled dataset show that the number of false positives/false negatives introduced by the learned prioir is small.





## Safe reconstruction



$$
\mathop{\text{argmin}}_x \parallel \phi(x) \parallel_1 \text{ s.t. } \parallel f(x) - y\parallel_2 < \epsilon \\
\mathop{\text{argmin}}_x \parallel \phi(x) \parallel_1 + \lambda \parallel f(x) - y\parallel_2 \\
$$

But what value of $\lambda$ should be chosen? it depends of $\epsilon$ and the amount of noise in the sampling process, $f$.

We want to pick $\lambda$ so that no progess on $\parallel f(x) - y \parallel_2$ is sacrificed.
But if you are too strict then the may end p reconstructing noise...!?
