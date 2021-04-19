from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
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

    from clusgan.models_cifar import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.utils import sample_z
    
except ImportError as e:
    print(e)
    raise ImportError

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    latent_dim = 50
    n_c = 10
    classifier =  torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)                 
    generator = Generator_CNN(50, 10, (3, 32, 32)).to(device) 
    encoder = Encoder_CNN(latent_dim, n_c).to(device) 

    encoder_dict = torch.load("C:\\Users\\BOBLY\\zzz\\final_project\\CSC413_Project\\clusterGAN-master\\runs\\cifar-10\\300epoch_z50_wass_bs64_wass_matric\\models\\encoder.pth.tar",
                                map_location=device)  
    generator_dict = torch.load("C:\\Users\\BOBLY\\zzz\\final_project\\CSC413_Project\\clusterGAN-master\\runs\\cifar-10\\300epoch_z50_wass_bs64_wass_matric\\models\\generator.pth.tar",
                                map_location=device) 

    generator.load_state_dict(generator_dict)
    encoder.load_state_dict(encoder_dict)

    if cuda:
        classifier.cuda()
        encoder.cuda()
        generator.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator.eval()
    encoder.eval()
    classifier.eval()

    # mode accuracy
    shape_constant = 1000
    correct = 0
    labels = {}
    for i in range(n_c):
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=shape_constant, latent_dim=latent_dim, n_c=n_c, fix_class=i)
        gen_imgs_samp = generator(zn_samp, zc_samp)
        output = classifier(gen_imgs_samp)
        generated = torch.argmax(output, 1)

        majority = torch.bincount(generated)
        label = torch.argmax(majority)
        count = torch.max(majority)
        correct += count
        labels[i] = label

    mode_accuracy = correct.item()/(n_c * shape_constant)
    print(f"mode accuracy: {mode_accuracy}\n")

    # reconstruction accuracy
    batch_size = 1000
    testdata = get_dataloader(dataset_name='cifar-10', train_set=False, batch_size=batch_size)
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = Variable(test_imgs.type(Tensor))

    t_imgs = test_imgs.data
    e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
    teg_imgs = generator(e_tzn, e_tzc)
    reconstruction_labels = torch.argmax(classifier(teg_imgs), 1)
    reconstruction_loss = torch.sum(torch.isclose(test_labels.cuda(), reconstruction_labels.cuda(), 0, 0, True)).item()/batch_size
    print(f"reconstruction loss: {reconstruction_loss}\n")

    # cluster accuracy
    encoder_result = torch.argmax(e_tzc_logits, 1)
    mapping = torch.LongTensor([labels[i] for i in encoder_result.tolist()])
    ACC = torch.sum(torch.isclose(test_labels.cuda(), mapping.cuda(), 0, 0, True)).item()/batch_size
    NMI = normalized_mutual_info_score(test_labels, mapping)
    ARI = adjusted_rand_score(test_labels, mapping)

    print(f"Cluster Accuracy: {ACC}\n")
    print(f"NMI: {NMI}\n")
    print(f"ARI: {ARI}\n")


if __name__ == "__main__":
    main()



