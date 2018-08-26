## Learned priors

Two alternatives were explored; VAEs, GANS with RIMs on the queue.

#### [GANs](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

I explored the properties of a simple 2d GAN. The generator takes a 1D noise signal and maps it into a 2D space. The goal is to learn to generate samples from P(x). However, the infamous mode-collapse-problem is present.

This is significant because if a GAN has dropped modes of the data then optimising $P_{gan}(x)$ will push $x$ towards only the modes captured by the GAN, and this may be at the cost of modes not captured by the GAN.

![pic](../assets/gan_init.png)
![pic](../assets/gan_collapse.png)

*At initialisation, the gradients don't point anywhere useful and the generator has been initialised spanning one of the modes.*

Unless the mode collapse problem can be reliably solved (with guarantees), GANs do not seem like a good idea. Note: there is a large amount of existing prior work here, see [reading.md](reading.md).

#### [VAEs](https://arxiv.org/abs/1312.6114)

I started with a VAE capable of producing good samples.

![pic](../assets/gen.png)

Then, to get an estimate of $P(x)$ I sampled from the posterior and used the prior to estimate its probability.

$$
\begin{align}
h_i &= f(x_i) \tag{encode $x_i$} \\
p(x) &:= \mathbb E_{z \sim p(\cdot | h_i)} \left[ p_{prior}(z) \right] \tag{expected prior prob}\\
\end{align}
$$

In the case of the latent space being define as a univariate gaussian, $h_i = \mu_i, \sigma_i$ and samples are taken from $\mathcal N(\mu_i, \sigma_i)$ and the prior is defined to be $\mathcal N(0, 1)$.

$$
\begin{align}
\hat x &= \mathop{\text{argmax}}_x p(x) \\
x_{t+1} &= x_t + \eta \frac{\partial p(x)}{\partial x} \tag{gradient ascent}\\
\end{align}
$$


So using this estimate of $P(x)$ we can follow its density estimate towards more likely images. Thus filling is any missing information in an reconstruction.


![pic](../assets/opt_px_same.png)
![pic](../assets/same_opt_px_imgs.png)

Despite and increase in P(x), there is no obvious difference in the images. Similarly, below...

![pic](../assets/p_x.png)
![pic](../assets/init_x.png)

*The $x_i$s at initialisation (initialised as MNIST digits plus noise).*

![pic](../assets/finals_x.png)

*The $x_i$s after 1000 steps of gradient ascent.*

So the general problem seems to be that we can happily optimise P(x) (examples below), but an increase in P(x) does not necessarily to give us useful results.

![pic](../assets/opt_px_graph_low_lr.png)
![pic](../assets/opt_px_low_lr.png)

*Images generated every 100 steps (left to right, top to bottom).*

![pic](../assets/opt_px.png)
![pic](../assets/opt_px_img.png)

*Images generated every 100 steps (left to right, top to bottom).*


***

Why doesnt this work??? Good question.

***

In hindsight I am not sure this makes sense. As $f(x)$ is supposed to give an approximation to the posterior distribution, $q(z | x)$
$$
\begin{align}
p(x) &= E_{z\sim p(z)} p(x \mid z)  \\
p(x | z) &= \frac{p(z \mid x)p(x)}{p(z)}  \\
p(x | z) &\approx \frac{q(z \mid x)p(x)}{p(z)}  \\
&\approx E_{x \sim D}  p(z \mid x)p(x) \\
\end{align}
$$


#### RIMs

The code for RIMs exists in [src](src), but it will take some more effort to reproduce the papers results.

#### Future

There is no reason R(x) must be explicitly representing ...
Could learn a sparse representation of the data
 $\mathop{\text{argmin}}_{\theta} \parallel d(e(x)) - x \parallel_2 + \parallel e(x) \parallel_1$ and use this to regularise reconstructions to plausible images.

Candidate reconstructions that are not sparse in this representation will be pushed in that direction.


## Density models

The deeper problem the learned prior approach is that we want to use the density of $X, P(x)$ to help us reconstruct images. Yet we dont have any good models of density...

![pic](../assets/generative_modelling_challenges.png)

#### Parzen windows

Given the tip above, I had a play with parzen windows to see how well they can be used to turn a VAE into a density estimate.

$$
p(x) = \frac{1}{n}\sum_{i=0}^n k(x, \hat x_i)
$$

So, I used the hidden space of a VAE as the inputs to a parzen window (hoping the the latent space would give greater generalisation as it is lower dimensional).

![pic](../assets/latent_space.png)

So using the equation above we can estimate the probability of data.
$$
p(x) = \frac{1}{n}\sum_{i=0}^n k(f(x), f(\hat x_i))
$$

![pic](../assets/parzen_high_temp.png)

Cool, we seem to have a nice structured estimate of P(x), with gradient pointing in meaningful directions.

But, using this parzen window we seem to get equal estiamtes of P(x) for images from MNIST and for generated (white noise) images.

![pic](../assets/p_parzen_fake_real.png)

Ok, the problem is that the kernels are too wide and are allocating probability to locations further away from the data. Solution, reduce the width of the kernels.

![pic](../assets/parzen_low_temp.png)
![pic](../assets/p_parzen_separated.png)

The problem is now that the gradients don't point in any meaningful directions. They will simply lead you to the nearest data point.

_Maybe there is an optimal tuning of the width to give desired results. But I didnt spend much time exploring this._

#### Future directions

Quick wins: Could try playing with [Gaussian processes](https://en.wikipedia.org/wiki/Gaussian_process) or [Neural processes](https://arxiv.org/abs/1807.01622).

Or look into information geometry to ...?