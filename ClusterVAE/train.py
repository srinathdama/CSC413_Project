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
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=1, type=int, help="Number of dataset workers")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    img_size = 28
    channels = 1
   
    # Latent space info
    latent_dim = 10
    betan = 10
    # betac = args.betac
   
    # wass_metric = args.wass_metric
    # mtype = 'van'
    # if (wass_metric):
    #     mtype = 'wass'
    
    mtype = 'vae_vanilla' 
    # Make directory structure for this run
    sep_und = '_'
    run_name_comps = ['%iepoch'%n_epochs, 'z%s'%str(latent_dim), mtype, 'bs%i'%batch_size, run_name]
    run_name = sep_und.join(run_name_comps)

    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
    
    x_shape = (channels, img_size, img_size)
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    decoder = Decoder_CNN(latent_dim, x_shape)
    encoder = Encoder_CNN(latent_dim)
    
    if cuda:
        decoder.cuda()
        encoder.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    dataloader = get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                num_workers=num_workers)

    # Test data loader
    testdata = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=test_batch_size, train_set=False)
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = Variable(test_imgs.type(Tensor))
   
    ge_chain = ichain(decoder.parameters(),
                      encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ----------
    #  Training
    # ----------
    ge_l = []
    mse_l = []
    kl_l =[]
    
    c_zn = []
    c_zc = []
    c_i = []
    
    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        for i, (imgs, itruth_label) in enumerate(dataloader):
           
            batchsize = imgs.shape[0]
            # Ensure generator/encoder are trainable
            decoder.train()
            encoder.train()
            # Zero gradients for models
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------------
            #  Train Encoder + Decoder
            # ---------------------------
            
            optimizer_GE.zero_grad()


            # Encode the generated images
            mu, sigma = encoder(real_imgs)
            
            # reparametrization trix 
            z = mu+torch.randn_like(mu)*sigma

            # Generate a batch of images
            gen_imgs = decoder(z)

            marginal_likelihood = -torch.pow(real_imgs - gen_imgs, 2).sum() / batchsize
            # print(marginal_likelihood2.item(), marginal_likelihood.item())
            KL_divergence = 0.5 * torch.sum(
                                        torch.pow(mu, 2) +
                                        torch.pow(sigma, 2) -
                                        torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                                    ).sum() / batchsize
    
            # calculte loss
            loss = -(marginal_likelihood - KL_divergence)

            loss.backward(retain_graph=True)
            optimizer_GE.step()

        # Save training losses
        ge_l.append(loss.item())
        mse_l.append(-marginal_likelihood.item())
        kl_l.append(KL_divergence.item())
        
   
    
    print('done training!')

    


    # Save training results
    train_df = pd.DataFrame({
                             'n_epochs' : n_epochs,
                             'learning_rate' : lr,
                             'beta_1' : b1,
                             'beta_2' : b2,
                             'weight_decay' : decay,
                             'latent_dim' : latent_dim,
                             'gen_enc_loss' : ['G+E', ge_l],
                             'mse_loss' : ['MSE', mse_l],
                             'kl_loss' : ['KL', kl_l]
                            })

    train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'mse_loss', 'kl_loss'],
                    figname='%s/training_model_losses.png'%(run_dir)
                    )

    # plot_train_loss(df=train_df,
    #                 arr_list=['zn_cycle_loss', 'zc_cycle_loss', 'img_cycle_loss'],
    #                 figname='%s/training_cycle_loss.png'%(run_dir)
    #                 )


    # Save current state of trained models
    model_list = [encoder, decoder]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
