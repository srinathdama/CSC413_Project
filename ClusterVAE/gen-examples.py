from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)

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

    from VAE.definitions import DATASETS_DIR, RUNS_DIR
    from VAE.CIFAR_models import CIFAR_Decoder_CNN, CIFAR_Encoder_CNN
    from VAE.models import Encoder_CNN, Decoder_CNN
    from VAE.datasets import get_dataloader, dataset_list
    from VAE.utils import sample_z

    from sklearn.manifold import TSNE
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Script to save generated examples from learned ClusterGAN generator")
    parser.add_argument("-r", "--run_dir", dest="run_dir", help="Training run directory")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=5000, type=int, help="Batch size")
    parser.add_argument('--ae', dest='vae_flag', action='store_false')
    parser.set_defaults(vae_flag=True)
    args = parser.parse_args()

    batch_size = args.batch_size
    vae_flag  = args.vae_flag
    
    # Directory structure for this run
    run_dir = args.run_dir.rstrip("/")
    run_name = run_dir.split(os.sep)[-1]
    dataset_name = run_dir.split(os.sep)[-2]
    
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')


    # Latent space info
    train_df = pd.read_csv('%s/training_details.csv'%(run_dir))
    latent_dim = train_df['latent_dim'][0]
    # n_c = train_df['n_classes'][0]
    n_c = 10

    if dataset_name == 'cifar10':
        img_size = 32
        channels = 3
    else:
        img_size = 28
        channels = 1
    
    x_shape = (channels, img_size, img_size)

    cuda = True if torch.cuda.is_available() else False
    

    if dataset_name == 'cifar10':
        decoder = CIFAR_Decoder_CNN(latent_dim, x_shape)
        encoder = CIFAR_Encoder_CNN(latent_dim, vae_flag)
    else:
        decoder = Decoder_CNN(latent_dim, x_shape)
        encoder = Encoder_CNN(latent_dim, vae_flag)

    # Load encoder model
    enc_fname = os.path.join(models_dir, encoder.name + '.pth.tar')
    encoder.load_state_dict(torch.load(enc_fname))
    encoder.cuda()
    encoder.eval()

    # Load generator model
    gen_fname = os.path.join(models_dir, decoder.name + '.pth.tar')
    decoder.load_state_dict(torch.load(gen_fname))
    decoder.cuda()
    decoder.eval()
   
   # Configure data loader
    dataloader = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=batch_size, train_set=False)
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Get full batch for encoding
    imgs, labels = next(iter(dataloader))
    c_imgs = Variable(imgs.type(Tensor), requires_grad=False)
    
    # Encode real images
    if vae_flag:
        enc_zmu, enc_sigma = encoder(c_imgs)
        z = enc_zmu + torch.randn_like(enc_zmu)*enc_sigma
    else:
        enc_zmu = encoder(c_imgs)
        z = enc_zmu

    # Loop through specific classes
    n_images = 100
    for idx in range(n_c):
        # zn, zc, zc_idx = sample_z(shape=batch_size, latent_dim=latent_dim, n_c=n_c, fix_class=idx, req_grad=False)
    
        indces = labels == idx

        # Generate a batch of images
        gen_imgs = decoder(z[indces])
        
        # Save some examples!
        save_image(gen_imgs.data[0:n_images], '%s/class_%i_gen.png' %(imgs_dir, idx), 
                   nrow=int(np.sqrt(n_images)), normalize=True)

        
        # Generate a batch of images
        gen_imgs = c_imgs[indces]
        
        # Save some examples!
        save_image(gen_imgs.data[0:n_images], '%s/class_input_%i.png' %(imgs_dir, idx), 
                   nrow=int(np.sqrt(n_images)), normalize=True)


if __name__ == "__main__":
    main()
