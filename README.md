# mri-reconstruction

This repo contains my notes and code while exploring how to use learned priors for compressed sensing.
The work was done in collaboration with Paul Teal (while I was working as a part-time research assistant).

## Docs

I have written a couple of documents detailing my work and thoughts on;

- [learned priors for comressed sensing](https://act65.github.io/mri-reconstruction/learned-prior)
- [density estimation](https://act65.github.io/mri-reconstruction/density)
- [reconstruction with guarantees](https://act65.github.io/mri-reconstruction/guarantees)

## Code

To install you will need to run

```
git clone https://github.com/act65/mri-reconstruction.git
cd mri-reconstruction
python setup.py install
```
You should then be able to run the scripts (to train new models). For example;
```
python scripts/train_infovae.py --logdir={path} \\
--n_hidden=4 --width=64 --epochs=100 --beta=1.0 \\
--learning_rate=0.0001 --batch_size=8
```

And to run the notebooks;
```
jupyter lab
```
and then navigate to [http://localhost:8888/lab](http://localhost:8888/lab).
