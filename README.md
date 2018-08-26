# mri-reconstruction

This repo contains my notes and code while exploring how to use learned priors for compressed sensing.

## Docs

I have written a couple of documents detailing my thoughts on;

- [learned priors for comressed sensing](https://github.com/act65/mri-reconstruction/blob/master/docs/learn-prior.md)
- [density estimation](https://github.com/act65/mri-reconstruction/blob/master/docs/density.md)
- [reconstruction with guarantees](https://github.com/act65/mri-reconstruction/blob/master/docs/guarantees.md)

## Code

To run the scripts you will need to

```
pip install tensorflow
cd mri-reconstruction
python setup.py install

python scripts/{script_name} --args
```
For example;
```
python scripts/train_infovae.py --logdir={path} \\
--n_hidden=4 --width=64 --epochs=100 --beta=1.0 \\
--learning_rate=0.0001 --batch_size=8
```


To run the notebooks you will need to
```
pip install jupyterlab
jupyter lab
```
and then navigate to [http://localhost:8888/lab](http://localhost:8888/lab).
