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
    from VAE.VaDE_model import VaDE
    from VAE.utils import save_model, sample_z, cross_entropy, run_clustering, run_vade_metrics
    from VAE.datasets import get_dataloader, dataset_list
    from VAE.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError


def lr_decay(global_step, init_learning_rate = 2e-4,
            min_learning_rate = 1e-4, decay_rate = 0.995):
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

    ## TO DO
    # pi_, mu_c, log_sigma2_c = 0,0,0
    npzfile_path = '/home/srinath/Project/CSC413_Project/GMVAE/runs/cifar10/400epoch_z50_vae_vanilla_bs128_BS_sensitivity/cifar10_GMM_10_gmm_weights.npz'
       
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
    
    arg_dict = {'x_shape': x_shape, 
                'nClusters': nClusters,
                'run_dir': run_dir,
                'models_dir': models_dir,
                'npzfile_path': npzfile_path,
                'cuda': cuda}
    
    vade = VaDE(args, arg_dict)

    # for name, param in vade.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if cuda:
        vade.cuda()
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


    optimizer_GE = torch.optim.Adam(vade.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
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
    
    # print_time = True
    
    ## load GMM weights 
    vade.pre_train(dataloader, 50)


    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        if print_time:
            time_epoch0 = datetime.now().timestamp()
        for i, (imgs, itruth_label) in enumerate(dataloader):
           
            batchsize = imgs.shape[0]
            # Ensure generator/encoder are trainable
            vade.train()
            # Zero gradients for models
            vade.zero_grad()
            
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------------
            #  Train Encoder + Decoder
            # ---------------------------
            
            optimizer_GE.zero_grad()
            
            loss, recon_error = vade.ELBO_Loss(real_imgs)


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
        vade.eval()
        t_imgs, t_label = test_imgs.data, test_labels

        loss, recon_error = vade.ELBO_Loss(t_imgs)
        
        
        if vae_flag:
            test_ge_l.append(loss.item())
            test_mse_l.append(recon_error.item())
        else:
            test_ge_l.append(loss.item())

        if print_time:
            delta_time_epoch = datetime.now().timestamp() - time_epoch0
            print('epoch :', epoch, ', time :', delta_time_epoch)
            print('train loss: ', ge_l[-1], ' recon loss: ', mse_l[-1])
            print('test loss: ', test_ge_l[-1], ' recon loss: ', test_ge_l[-1])
   
    
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
    model_list = [vade]
    save_model(models=model_list, out_dir=models_dir)


    ### accuracy
    train_pred_label = []
    train_label = []
    test_pred_label = []
    test_label = []
    
    batch_size = 5000
    traindataloader =  get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                train_set=True)

    for i, (imgs, itruth_label) in enumerate(traindataloader):
        train_imgs = Variable(imgs.type(Tensor))
        with torch.no_grad():
            vade_label  = vade.predict(train_imgs)
        train_pred_label.append(vade_label)
        train_label.append(itruth_label)
    train_pred_label = np.concatenate(train_pred_label, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_data = [train_pred_label, train_label]

    testdataloader =  get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                train_set=False)

    for i, (imgs, itruth_label) in enumerate(testdataloader):
        train_imgs = Variable(imgs.type(Tensor))
        with torch.no_grad():
            vade_label  = vade.predict(train_imgs)
        test_pred_label.append(vade_label)
        test_label.append(itruth_label)
    test_pred_label = np.concatenate(test_pred_label, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    test_data = [test_pred_label, test_label]

    noof_clusters = nClusters
    target_names  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    save_path     = run_dir
    run_vade_metrics(noof_clusters, target_names, save_path, dataset_name, train_data, test_data)

    



if __name__ == "__main__":
    main()
