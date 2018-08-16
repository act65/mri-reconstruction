# mri-reconstruction

* Reconstruction with context (coronal, sagital, ventral sections or patient history, genetics)
* Reconstruction with few samples (learned priors, sparse bases, compressed sampling)
* (Learn to) iteratively pick the best points to samples that minimise uncertainty and maximise the accuracy of reconstruction.  


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
cd mri-reconstruction
jupyter lab
```
and then navigate to [http://localhost:8888/lab](http://localhost:8888/lab).
