from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
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

    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.models import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.plots import plot_train_loss
    from classifier import Net
except ImportError as e:
    print(e)
    raise ImportError

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classifier = Net().to(device)                 
classifier.load_state_dict(torch.load("mnist_cnn.pt"))

generator = Generator_CNN(30, 10, (1, 28, 28)).to(device) 
encoder = Encoder_CNN(30, 10).to(device) 

encoder_dict = torch.load("C:\\Users\\BOBLY\\zzz\\final_project\\CSC413_Project\\clusterGAN-master\\runs\\mnist\\500epoch_z30_van_bs64_test_run\\models\\encoder.pth.tar",
                            map_location=device)  
generator_dict = torch.load("C:\\Users\\BOBLY\\zzz\\final_project\\CSC413_Project\\clusterGAN-master\\runs\\mnist\\500epoch_z30_van_bs64_test_run\\models\\generator.pth.tar",
                            map_location=device) 

generator.load_state_dict(generator_dict)
encoder.load_state_dict(encoder_dict)

generator.eval()
encoder.eval()
classifier.eval()

# mode accuracy
zn_samp, zc_samp, zc_samp_idx = sample_z(shape=1000, latent_dim=30, n_c=10, fix_class=0)
gen_imgs_samp = generator(zn_samp, zc_samp)
output = classifier(gen_imgs_samp)
print(zn_samp.requires_grad)
generated = torch.argmax(output, 1)
print(generated)
print(zc_samp_idx)

# reconstruction accuracy
# testdata = get_dataloader(train_set=False)
# test_imgs, test_labels = next(iter(testdata)).cpu()
# test_imgs = Variable(test_imgs.type(Tensor))
# t_imgs, t_label = test_imgs.data, test_labels
# e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
# teg_imgs = generator(e_tzn, e_tzc)
# teg.requires_grad = False
# reconstruction_labels = torch.argmax(classifier(teg_imgs), 1).cpu()
# print(test_labels - reconstruction_labels)


