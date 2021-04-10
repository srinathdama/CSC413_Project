from __future__ import print_function

try:
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

    from VAE.utils import tlog, softmax, initialize_weights, calc_gradient_penalty
except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )


class CIFAR_Decoder_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, x_shape, verbose=False):
        super(CIFAR_Decoder_CNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.x_shape = x_shape
        self.ishape = (448, 2, 2)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim , self.iels),
            nn.BatchNorm1d(self.iels),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
        
            # Reshape to 448 x (2x2)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.ConvTranspose2d(448, 256, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),

            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)
    
    def forward(self, z):
        #z = z.unsqueeze(2).unsqueeze(3)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class CIFAR_Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, VAE=True, verbose=False):
        super(CIFAR_Encoder_CNN, self).__init__()

        self.name = 'encoder'
        self.channels = 3
        self.latent_dim = latent_dim
        self.cshape = (512, 2, 2)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        self.vae_flag = VAE
        if self.vae_flag:
            self.out_dim = 2*latent_dim
        else:
            self.out_dim = latent_dim
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, self.out_dim)
        )

        initialize_weights(self)
        
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)

        if self.vae_flag:
            # Separate mu and sigma
            mu = z[:, 0:self.latent_dim]

            # ensure sigma is postive 
            sigma = 1e-6 + F.softplus(z[:, self.latent_dim:])
            
            return [mu, sigma]
        else:
            return z


class CIFAR_SEncoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, VAE=True, verbose=False):
        super(CIFAR_SEncoder_CNN, self).__init__()

        self.name = 'encoder'
        self.channels = 3
        self.latent_dim = latent_dim
        self.cshape = (128, 6, 6)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        self.vae_flag = VAE
        if self.vae_flag:
            self.out_dim = 2*latent_dim
        else:
            self.out_dim = latent_dim
        
        # self.model = nn.Sequential(
        #     # Convolutional layers
        #     nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, 4, stride=2, bias=True),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     # Flatten
        #     Reshape(self.lshape),
            
        #     # Fully connected layers
        #     torch.nn.Linear(self.iels, 1024),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     torch.nn.Linear(1024, self.out_dim)
        # )

        # https://github.com/1Konny/Beta-VAE/blob/master/model.py
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64, 8, 8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64, 4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            Reshape((256*1*1,)),                 # B, 256
            nn.Linear(256, self.out_dim)         # B, z_dim*2)
            )             

        initialize_weights(self)
        
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)

        if self.vae_flag:
            # Separate mu and sigma
            mu = z[:, 0:self.latent_dim]

            # ensure sigma is postive 
            sigma = 1e-6 + F.softplus(z[:, self.latent_dim:])
            
            return [mu, sigma]
        else:
            return z
        
class CIFAR_SDecoder_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterVAE
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC8x8x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, x_shape, verbose=False):
        super(CIFAR_SDecoder_CNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.x_shape = x_shape
        self.ishape = (128, 8, 8)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        
        # self.model = nn.Sequential(
        #     # Fully connected layers
        #     torch.nn.Linear(self.latent_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     #torch.nn.ReLU(True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     torch.nn.Linear(1024, self.iels),
        #     nn.BatchNorm1d(self.iels),
        #     #torch.nn.ReLU(True),
        #     nn.LeakyReLU(0.2, inplace=True),
        
        #     # Reshape to 128 x (8x8)
        #     Reshape(self.ishape),

        #     # Upconvolution layers
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
        #     nn.BatchNorm2d(64),
        #     #torch.nn.ReLU(True),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True),
        #     nn.Sigmoid()
        # )
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),               # B, 256
            Reshape((256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), # B,  3, 32, 32
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)
    
    def forward(self, z):
        #z = z.unsqueeze(2).unsqueeze(3)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen
