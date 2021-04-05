from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import pandas as pd
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score

    from VAE.definitions import DATASETS_DIR, RUNS_DIR
    from VAE.models import Decoder_CNN, Encoder_CNN
    from VAE.utils import save_model, sample_z, cross_entropy, run_clustering
    from VAE.datasets import get_dataloader, dataset_list
    from VAE.plots import plot_train_loss

except ImportError as e:
    print(e)
    raise ImportError

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
             
    generator = Decoder_CNN(10, (1, 28, 28)).to(device) 
    encoder = Encoder_CNN(10).to(device) 

    dir_path = '/home/srinath/Project/CSC413_Project/ClusterVAE/runs/mnist/200epoch_z10_vae_vanilla_bs256_test_run/'
    encoder_dict = torch.load(os.path.join(dir_path, "models/encoder.pth.tar"),
                                map_location=device)  
    generator_dict = torch.load(os.path.join(dir_path, "models/generator.pth.tar"),
                                map_location=device) 

    generator.load_state_dict(generator_dict)
    encoder.load_state_dict(encoder_dict)

    if cuda:
        encoder.cuda()
        generator.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator.eval()
    encoder.eval()

    # mode accuracy
    num_classes = 10
    shape_constant = 1000
    correct = 0
    labels = {}


    ## train GMM model 
    mu_train = []
    train_label = []
    mu_test = []
    test_label = []
    
    batch_size = 5000
    traindataloader = get_dataloader(train_set=True, batch_size=batch_size)
    for i, (imgs, itruth_label) in enumerate(traindataloader):
        train_imgs = Variable(imgs.type(Tensor))
        with torch.no_grad():
            mu, sigma  = encoder(train_imgs)
        mu_train.append(mu.cpu().numpy())
        train_label.append(itruth_label)
    mu_train = np.concatenate(mu_train, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_data = [mu_train, train_label]

    testdataloader = get_dataloader(train_set=False, batch_size=batch_size)
    for i, (imgs, itruth_label) in enumerate(testdataloader):
        train_imgs = Variable(imgs.type(Tensor))
        with torch.no_grad():
            mu, sigma  = encoder(train_imgs)
        mu_test.append(mu.cpu().numpy())
        test_label.append(itruth_label)
    mu_test = np.concatenate(mu_test, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    test_data = [mu_test, test_label]
    
    noof_clusters = 10
    target_names  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    save_path     = dir_path
    data_set      = 'MNIST'
    run_clustering('Kmeans', noof_clusters, target_names, save_path, data_set, train_data, test_data)

    run_clustering('GMM', noof_clusters, target_names, save_path, data_set, train_data, test_data)


if __name__ == "__main__":
    main()



