## Learning the inverse

There exists $f: x \rightarrow y$, want $f^{-1}$. In all cases we get acess to y.

| X  | Sparsity  | no f  | f |
|---|---|---|---|
| x | sparse x | dictionary learning!?   |   |
| no x | sparse x |  clustering __X__ | compressed sensing  |
| x |  sparse y | supervised learning (categorical) | __us__   |
| no x | sparse y | __X__ |  also compressed sensing __?__  |
| x |  not sparse | supervised learning (regression)  |   |
| no x | not sparse  | unsupervised learning __X__ |   |

_also a distinction between having X versus having pairs of (X, Y)!?_

__X__ indicates that $f^{-1}$ is not recoverable


__Q__ What does having $f$, or X, or ... buy us? Want bounds on each for thair ability to efficiently learn $f^{-1}$.


Properties of $f$

| X  | Sparsity  | no noise  | noise |
|---|---|---|---|
| linear | invertible |   |   |
| non-linear | invertible |   |   |
| linear | not invertible |   |   |
| non-linear | not invertible |   |   |

Two parts.
- __Consistency__: Those x's that are consistent with the observations, y.
- __Search__: Of the possible x's, which ones are plausible given other assumtions and knowledge!?

Ideally we would only consider consistent x images?
- How can we ensure this?
- How can we find the minimal set that is consistent?
- ?



## Setting

Given $(X, Y, f, df)$, we want $f^{-1}$.

- $f$ is lossy.
-

$$
\begin{align}
x &\neq f^{-1}(f(x)) \tag{not invertible/reversible}\\
x^* &= \mathop{\text{argmin}}_x d(y, f(x)) + R(x) \tag{optimisation problem} \\
\end{align}
$$

Point is that $d$ is an imperfect measure because of $f$ (which throws away info). So we need $R(x)$ to incorporate prior about likely values of $x$.

### Lossy

The loss of info. This means $f: x \rightarrow y$ is a many-to-one function.

E.g. Given $x_1, x_2$, let $y_1 = f(x_1) = y_2 = f(x_2)$, thus, no function can take $y_1$ as input and recover the true input.

Thus there exists no function
