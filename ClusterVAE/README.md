
## Run ClusterVAE on MNIST

To run ClusterVAE on the MNIST dataset, ensure the package is setup and then run
```
python train.py -r test_run -s mnist -b 256 -n 300
```
where a directory `runs/mnist/test_run` will be made and contain the generated output
(models, example generated instances, training figures) from the training run.
The `-r` option denotes the run name, `-s` the dataset (currently MNIST and Fashion-MNIST),
`-b` the batch size, and `-n` the number of training epochs.


Run criteria.py file to load the trained model and train GMM/Kmeans models on the latent embedding vector. Results will be saved to `runs/mnist/test_run`
```
python criteria.py
```