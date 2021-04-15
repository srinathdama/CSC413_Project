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
    from VAE.CIFAR_models import CIFAR_Decoder_CNN, CIFAR_Encoder_CNN, CIFAR_SDecoder_CNN, CIFAR_SEncoder_CNN
    from VAE.utils import save_model, sample_z, cross_entropy, run_clustering
    from VAE.datasets import get_dataloader, dataset_list
    from VAE.plots import plot_train_loss

except ImportError as e:
    print(e)
    raise ImportError

def main():

    global args
    parser = argparse.ArgumentParser(description="Script to save generated examples from learned ClusterGAN generator")
    parser.add_argument("-r", "--run_dir", dest="run_dir", help="Training run directory")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=5000, type=int, help="Batch size")
    parser.add_argument('--ae', dest='vae_flag', action='store_false')
    parser.add_argument('--c', dest='classifier_flag', action='store_false')
    parser.add_argument('--cifar_big_arch', dest='cifar_big_arch', action='store_true')
    parser.set_defaults(vae_flag=True)
    parser.set_defaults(classifier_flag=False)
    parser.set_defaults(cifar_big_arch=False)
    args = parser.parse_args()

    batch_size = args.batch_size
    vae_flag  = args.vae_flag
    classifier_flag = args.classifier_flag
    cifar_big_arch = args.cifar_big_arch
    
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False

    vae_flag  = args.vae_flag

    if vae_flag:
        mtype = 'vae_vanilla' 
    else:
        mtype = 'ae' 

    # run_name = args.run_name
    # dataset_name = args.dataset_name
    if dataset_name == 'cifar10':
        img_size = 32
        channels = 3
    else:
        img_size = 28
        channels = 1
        classifier_flag = False


    # Training details
    # n_epochs = args.n_epochs
    # batch_size = args.batch_size

    # latent_dim = args.latent_dim

    x_shape = (channels, img_size, img_size)

    # Initialize generator and discriminator
    if dataset_name == 'cifar10':
        # cifar_big_arch = True
        if cifar_big_arch:
            generator = CIFAR_Decoder_CNN(latent_dim, x_shape).to(device)
            encoder = CIFAR_Encoder_CNN(latent_dim, vae_flag).to(device)
        else:
            generator = CIFAR_SDecoder_CNN(latent_dim, x_shape).to(device)
            encoder = CIFAR_SEncoder_CNN(latent_dim, vae_flag).to(device)
    else:
        generator = Decoder_CNN(latent_dim, x_shape).to(device) 
        encoder = Encoder_CNN(latent_dim, vae_flag).to(device) 

    # # Make directory structure for this run
    # sep_und = '_'
    # run_name_comps = ['%iepoch'%n_epochs, 'z%s'%str(latent_dim), mtype, 'bs%i'%batch_size, run_name]
    # run_name = sep_und.join(run_name_comps)

    # run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    # data_dir = os.path.join(DATASETS_DIR, dataset_name)
    # imgs_dir = os.path.join(run_dir, 'images')
    # models_dir = os.path.join(run_dir, 'models')

    # os.makedirs(data_dir, exist_ok=True)
    # os.makedirs(run_dir, exist_ok=True)
    # os.makedirs(imgs_dir, exist_ok=True)
    # os.makedirs(models_dir, exist_ok=True)

    # dir_path = '/home/srinath/Project/CSC413_Project/ClusterVAE/runs/mnist/200epoch_z10_vae_vanilla_bs256_test_run/'
    dir_path = run_dir
    encoder_dict = torch.load(os.path.join(dir_path, "models/encoder.pth.tar"),
                                map_location=device)  
    generator_dict = torch.load(os.path.join(dir_path, "models/generator.pth.tar"),
                                map_location=device) 

    generator.load_state_dict(generator_dict)
    encoder.load_state_dict(encoder_dict)
    # classifier =  torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)

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

    def predict(self,x):
        z_mu, z_sigma2_log = encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    ## train GMM model 
    mu_train = []
    train_label = []
    mu_test = []
    test_label = []
    
    batch_size = 5000
    traindataloader =   get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                train_set=True)
    for i, (imgs, itruth_label) in enumerate(traindataloader):
        train_imgs = Variable(imgs.type(Tensor))
        with torch.no_grad():
            if vae_flag:
                [mu, sigma]  = encoder(train_imgs)
            else:
                mu = encoder(train_imgs)
        mu_train.append(mu.cpu().numpy())
        train_label.append(itruth_label)
    mu_train = np.concatenate(mu_train, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_data = [mu_train, train_label]

    testdataloader =  get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                train_set=False)
    for i, (imgs, itruth_label) in enumerate(testdataloader):
        train_imgs = Variable(imgs.type(Tensor))
        with torch.no_grad():
            if vae_flag:
                [mu, sigma]  = encoder(train_imgs)
            else:
                mu = encoder(train_imgs)
        mu_test.append(mu.cpu().numpy())
        test_label.append(itruth_label)
    mu_test = np.concatenate(mu_test, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    test_data = [mu_test, test_label]
    
    noof_clusters = 10
    target_names  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    save_path     = dir_path
    # data_set      = 'MNIST'
    run_clustering('Kmeans', noof_clusters, target_names, save_path, dataset_name, train_data, test_data)

    run_clustering('GMM', noof_clusters, target_names, save_path, dataset_name, train_data, test_data)


if __name__ == "__main__":
    main()



