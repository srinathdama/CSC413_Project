from __future__ import print_function

try:
    import numpy as np
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
except ImportError as e:
    print(e)
    raise ImportError


DATASET_FN_DICT = {'mnist' : datasets.MNIST,
                   'fashion-mnist' : datasets.FashionMNIST,
                   'cifar-10' : datasets.CIFAR10
                  }


dataset_list = DATASET_FN_DICT.keys()


def get_dataset(dataset_name='mnist'):
    """
    Convenience function for retrieving
    allowed datasets.
    Parameters
    ----------
    name : {'mnist', 'fashion-mnist'}
          Name of dataset
    Returns
    -------
    fn : function
         PyTorch dataset
    """
    if dataset_name in DATASET_FN_DICT:
        fn = DATASET_FN_DICT[dataset_name]
        return fn
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, DATASET_FN_DICT.keys()))



def get_dataloader(dataset_name='mnist', data_dir='', batch_size=64, train_set=True, num_workers=1):

    dset = get_dataset(dataset_name)
    path = "C:\\Users\\BOBLY\\zzz\\MNIST"
    if dataset_name == 'cifar-10':
        path = "C:\\Users\\BOBLY\\zzz\\cifar-10"
    elif dataset_name == 'fashion-mnist':
        path = "C:\\Users\\BOBLY\\zzz\\Fashion-MNIST"
    else:
        path= "C:\\Users\\BOBLY\\zzz\\MNIST"

    dataloader = torch.utils.data.DataLoader(
        dset(path, train=train_set, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                       ])),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True)

    return dataloader
