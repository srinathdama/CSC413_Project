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
    from tqdm import tqdm
    from torch.optim import Adam
    import itertools
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import accuracy_score
    from itertools import chain as ichain
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score

    from VAE.definitions import DATASETS_DIR, RUNS_DIR
    from VAE.models import Decoder_CNN, Encoder_CNN
    from VAE.CIFAR_models import CIFAR_Decoder_CNN, CIFAR_Encoder_CNN, CIFAR_SDecoder_CNN, CIFAR_SEncoder_CNN
    # from VAE.VaDE_model import VaDE
    from VAE.utils import save_model, sample_z, cross_entropy, run_clustering
    from VAE.datasets import get_dataloader, dataset_list
    from VAE.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError

def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


class VaDE(nn.Module):
    def __init__(self,args, arg_dict):
        super(VaDE,self).__init__()

        self.name = 'VaDE'
        # Initialize generator and discriminator
        if args.dataset_name == 'cifar10':
            if args.cifar_big_arch:
                self.decoder = CIFAR_Decoder_CNN(args.latent_dim, arg_dict['x_shape'])
                self.encoder = CIFAR_Encoder_CNN(args.latent_dim, args.vae_flag)
            else:
                self.decoder = CIFAR_SDecoder_CNN(args.latent_dim, arg_dict['x_shape'])
                self.encoder = CIFAR_SEncoder_CNN(args.latent_dim, args.vae_flag)
        else:
            self.decoder = Decoder_CNN(args.latent_dim, arg_dict['x_shape'])
            self.encoder = Encoder_CNN(args.latent_dim, args.vae_flag)

        self.pi_=nn.Parameter(torch.FloatTensor(arg_dict['nClusters'],).fill_(1)/arg_dict['nClusters'],requires_grad=False)
        self.mu_c=nn.Parameter(torch.FloatTensor(arg_dict['nClusters'],args.latent_dim).fill_(0),requires_grad=False)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(arg_dict['nClusters'],args.latent_dim).fill_(0),requires_grad=False)


        self.args=args
        self.arg_dict = arg_dict


    def pre_train(self, dataloader= None, pre_epoch=10):

        model_file = os.path.join(self.arg_dict['models_dir'], 'pretrain_model.pk')
        if  not os.path.exists(model_file):
            
            npzfile = np.load(self.arg_dict['npzfile_path'])
            weights_np  = npzfile['weights']
            means_np  = npzfile['means']
            covariances_np  = npzfile['covariances']

            self.pi_.data = torch.from_numpy(weights_np).cuda().float()
            self.mu_c.data = torch.from_numpy(means_np).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(covariances_np).cuda().float())
            # Loss=nn.MSELoss()
            # opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            # print('Pretraining......')
            # epoch_bar=tqdm(range(pre_epoch))
            # for _ in epoch_bar:
            #     L=0
            #     for x,y in dataloader:
            #         if self.arg_dict['cuda']:
            #             x=x.cuda()

            #         z,_=self.encoder(x)
            #         x_=self.decoder(z)
            #         loss=Loss(x,x_)

            #         L+=loss.detach().cpu().numpy()

            #         opti.zero_grad()
            #         loss.backward()
            #         opti.step()

            #     epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

            # self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            # Z = []
            # Y = []
            # with torch.no_grad():
            #     for x, y in dataloader:
            #         if self.arg_dict['cuda']:
            #             x = x.cuda()

            #         z1, z2 = self.encoder(x)
            #         assert F.mse_loss(z1, z2) == 0
            #         Z.append(z1)
            #         Y.append(y)

            # Z = torch.cat(Z, 0).detach().cpu().numpy()
            # Y = torch.cat(Y, 0).detach().numpy()

            # gmm = GaussianMixture(n_components=self.arg_dict['nClusters'], covariance_type='diag')

            # pre = gmm.fit_predict(Z)
            # print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            # self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            # self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            # self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

            torch.save(self.state_dict(), model_file)

        else:

            self.load_state_dict(torch.load(model_file))


    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    def ELBO_Loss(self,x,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro= self.decoder(z)

            L_rec+= torch.pow(x - x_pro, 2).sum() / x.shape[0]

            # L_rec+=F.binary_cross_entropy(x_pro,x)

        L_rec/=L

        if torch.isnan(L_rec):
            print( L_rec.item())

        Loss=L_rec

        pi=self.pi_
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))

        if torch.isnan(Loss) or torch.isnan(L_rec):
            print(Loss.item(), L_rec.item())

        return Loss, L_rec



    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.arg_dict['nClusters']):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)


    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))