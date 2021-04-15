from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    from datetime import datetime

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


def lr_decay(global_step, init_learning_rate = 2e-4,
            min_learning_rate = 5e-5, decay_rate = 0.99):
    lr = ((init_learning_rate - min_learning_rate) *
          pow(decay_rate, global_step) +
          min_learning_rate)
    return lr

def main():
    startTime = datetime.now().timestamp()

    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=300, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=50, type=int, help="latent dimension")
    parser.add_argument("-v", "--beta_vae", dest="beta_vae", default=1, type=int, help="beta vae")
    parser.add_argument("-l", "--lr", dest="lr", default=4e-3, type=float, help="learning rate")
    parser.add_argument("--sigma_scale", dest="sigma_scale", default=1, type=float, help="sigma_scale")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=4, type=int, help="Number of dataset workers")
    parser.add_argument('--lr_decay_f', dest='lr_decay_f', action='store_true')
    parser.add_argument('--ae', dest='vae_flag', action='store_false')
    parser.add_argument('--cifar_big_arch', dest='cifar_big_arch', action='store_true')
    parser.add_argument('--print_time', dest='print_time', action='store_true')
    parser.set_defaults(vae_flag=True)
    parser.set_defaults(lr_decay_f=False)
    parser.set_defaults(cifar_big_arch=False)
    parser.set_defaults(print_time=False)
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers
    vae_flag  = args.vae_flag
    print_time = args.print_time
    beta_vae   = args.beta_vae
    cifar_big_arch = args.cifar_big_arch
    sigma_scale = args.sigma_scale
    lr_decay_f  = args.lr_decay_f

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr0 = args.lr
    if lr_decay_f:
        lr = lr_decay(0, lr0)
    else:
        lr  = lr0
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5  # 2.0*1e-6
    n_skip_iter = 1 #5

    if dataset_name == 'cifar10':
        img_size = 32
        channels = 3
    else:
        img_size = 28
        channels = 1
    
    # Latent space info
    latent_dim = args.latent_dim
    betan = 10
    nClusters = 10
    # betac = args.betac
   
    # wass_metric = args.wass_metric
    # mtype = 'van'
    # if (wass_metric):
    #     mtype = 'wass'
    
    if vae_flag:
        mtype = 'vae_vanilla' 
    else:
        mtype = 'ae' 
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
    bce_loss = torch.nn.BCELoss(reduction='sum')
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    if dataset_name == 'cifar10':
        if cifar_big_arch:
            decoder = CIFAR_Decoder_CNN(latent_dim, x_shape)
            encoder = CIFAR_Encoder_CNN(latent_dim, vae_flag)
        else:
            decoder = CIFAR_SDecoder_CNN(latent_dim, x_shape)
            encoder = CIFAR_SEncoder_CNN(latent_dim, vae_flag)
    else:
        decoder = Decoder_CNN(latent_dim, x_shape)
        encoder = Encoder_CNN(latent_dim, vae_flag)
    
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
    # https://discuss.pytorch.org/t/learning-rate-decay-during-training/74017
    if lr_decay_f:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_GE,
                    lr_lambda=lambda epoch: lr_decay(epoch, lr0)/lr0)
    # ----------
    #  Training
    # ----------
    ge_l = []
    mse_l = []
    kl_l =[]

    test_ge_l = []
    test_mse_l = []
    test_kl_l =[]
    # bse_loss = nn.BCELoss(reduction='sum')

    pi_=nn.Parameter(torch.FloatTensor(nClusters,).fill_(1)/nClusters,requires_grad=True).cuda()
    mu_c=nn.Parameter(torch.FloatTensor(nClusters, latent_dim).fill_(0),requires_grad=True).cuda()
    log_sigma2_c=nn.Parameter(torch.FloatTensor(nClusters, latent_dim).fill_(0),requires_grad=True).cuda()
    
    # print_time = True
    def ELBO_Loss(x, pi_, mu_c, log_sigma2_c,  L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = encoder(x)
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro= decoder(z)

            L_rec+= torch.pow(x - x_pro, 2).sum() / x.shape[0]

        L_rec/=L

        Loss=L_rec

        pi= pi_
        log_sigma2_c= log_sigma2_c
        mu_c= mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+ gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss, L_rec


    def gaussian_pdfs_log(x,mus,log_sigma2s):
        G=[]
        for c in range(nClusters):
            G.append(gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)



    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))
    
    ## load GMM weights 

    ## TO DO
    # pi_, mu_c, log_sigma2_c = 0,0,0
    npzfile = np.load('/home/srinath/Project/CSC413_Project/GMVAE/runs/cifar10/400epoch_z50_vae_vanilla_bs128_BS_sensitivity/cifar10_GMM_10_gmm_weights.npz')
    weights_np  = npzfile['weights']
    means_np  = npzfile['means']
    covariances_np  = npzfile['covariances']

    pi_.data = torch.from_numpy(weights_np).cuda().float()
    mu_c.data = torch.from_numpy(means_np).cuda().float()
    log_sigma2_c.data = torch.log(torch.from_numpy(covariances_np).cuda().float())

    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        if print_time:
            time_epoch0 = datetime.now().timestamp()
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
            
            loss, recon_error = ELBO_Loss(real_imgs, pi_, mu_c, log_sigma2_c)

            loss.backward(retain_graph=True)
            optimizer_GE.step()

        if lr_decay_f:
            scheduler.step()
            # print(epoch, optimizer_GE.param_groups[0]['lr'], lr_decay(epoch, lr0))
            # assert optimizer_GE.param_groups[0]['lr'] == lr_decay(epoch)
            

        # Save training losses
        # print(pi_)
        if vae_flag:
            ge_l.append(loss.item())
            mse_l.append(recon_error.item())
        else:
            ge_l.append(loss.item())

        # Generator in eval mode
        decoder.eval()
        encoder.eval()
        t_imgs, t_label = test_imgs.data, test_labels

        loss, recon_error = ELBO_Loss(t_imgs, pi_, mu_c, log_sigma2_c)
        
        
        if vae_flag:
            test_ge_l.append(loss.item())
            test_mse_l.append(recon_error.item())
        else:
            test_ge_l.append(loss.item())

        if print_time:
            delta_time_epoch = datetime.now().timestamp() - time_epoch0
            print('epoch :', epoch, ', time :', delta_time_epoch)
   
    
    print('done training!')

    loop_time = (datetime.now().timestamp() - startTime)

    print('total time :', loop_time)

    # Save training results
    if vae_flag:
        train_df = pd.DataFrame({
                                'n_epochs' : n_epochs,
                                'learning_rate' : lr0,
                                'lr_decay_f': lr_decay_f,
                                'beta_1' : b1,
                                'beta_2' : b2,
                                'weight_decay' : decay,
                                'latent_dim' : latent_dim,
                                'cifar_big_arch' : cifar_big_arch,
                                'sigma_scale' : sigma_scale,
                                'beta_vae' : beta_vae,
                                'gen_enc_loss' : ['G+E', ge_l],
                                'mse_loss' : ['MSE', mse_l],
                                'test_gen_enc_loss' : ['G+E test', test_ge_l],
                                'test_mse_loss' : ['MSE test', test_mse_l],
                                'time' : loop_time
                                })
    else:
        train_df = pd.DataFrame({
                            'n_epochs' : n_epochs,
                            'learning_rate' : lr0,
                            'lr_decay_f': lr_decay_f,
                            'beta_1' : b1,
                            'beta_2' : b2,
                            'weight_decay' : decay,
                            'latent_dim' : latent_dim,
                            'cifar_big_arch' : cifar_big_arch,
                            'sigma_scale' : sigma_scale,
                            'beta_vae' : beta_vae,
                            'gen_enc_loss' : ['MSE', ge_l],
                            'test_gen_enc_loss' : ['MSE test', test_ge_l],
                            'time' : loop_time
                        })

    train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    if vae_flag:
        plot_train_loss(df=train_df,
                        arr_list=['gen_enc_loss', 'mse_loss',
                         'test_gen_enc_loss', 'test_mse_loss'],
                        figname='%s/training_model_losses.png'%(run_dir)
                        )
    else:
        plot_train_loss(df=train_df,
                        arr_list=['gen_enc_loss', 'test_gen_enc_loss'],
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
