# ClusterGAN: A PyTorch Implementation

This is a PyTorch implementation of [ClusterGAN](https://arxiv.org/abs/1809.03627),
an approach to unsupervised clustering using generative adversarial networks.


## Requirements

The package as well as the necessary requirements can be installed by running `make` or via
```
virtualenv -p /usr/local/bin/python3 venv
source venv/bin/activate
python setup.py install
```

## Run ClusterGAN on MNIST

To run ClusterGAN on the MNIST dataset, ensure the package is setup and then run
```
python train.py -r test_run -s mnist -b 64 -n 300
```
where a directory `runs/mnist/test_run` will be made and contain the generated output
(models, example generated instances, training figures) from the training run.
The `-r` option denotes the run name, `-s` the dataset (currently MNIST and Fashion-MNIST),
`-b` the batch size, and `-n` the number of training epochs.

```
python gen-examples -r <directory of your saved model>  -b <batchsize>
```

```
python tsne-cluster.py -r <directory of your saved model> -n <number of samples>
```
```
python criteria.py
```

## Referenes

https://github.com/zhampel/clusterGAN
