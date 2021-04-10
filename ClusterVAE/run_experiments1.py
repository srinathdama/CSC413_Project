import subprocess
import os
import numpy as np

# subprocess.Popen('python /home/srinath/Project/CSC413_Project/ClusterVAE/train.py -r test_automate -b 64 -d 10 -s mnist', shell=False)

# latent_dims = [30, 50]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     os.system('python train.py -r latent_dem_sensitivity -b 64 -d '+str(latent_dim) + ' -s mnist')  


# batch_sizes = [32, 64, 128, 256, 1024, 4096]
# lrs         = 1e-4* np.asarray(batch_sizes, dtype= float)/batch_sizes[0]
# print(lrs)

# print(batch_sizes[3:], lrs[3:])
# for batch_size, lr in zip(batch_sizes[3:], lrs[3:]):
#     subprocess.run('python train.py -r BS_sensitivity -b '+ str(batch_size) + ' -l ' + str(lr) + ' -d 10 -s mnist', shell=True)


## CIFAR10
# batch_sizes = np.array([64, 128, 256, 1024])
# lrs         = 1e-4* np.asarray(batch_sizes, dtype= float)/batch_sizes[0]
# print(lrs)

# print(batch_sizes[[1,2]], lrs[[1,2]])
# for batch_size, lr in zip(batch_sizes[[1,2]], lrs[[1,2]]):
#     subprocess.run('python train.py -n 400 -r BS_sensitivity -b '+ str(batch_size) + ' -l ' + str(lr) + ' -d 50 -s cifar10', shell=True)


# latent_dims = [30, 50]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     subprocess.run('python train.py -n 400 -r latent_dem_sensitivity -b 64 -d '+str(latent_dim) + ' -s cifar10 ', shell=True)
    

betas = [2, 3, 4, 5]
# latent_dims = [5]
for beta in betas:
    subprocess.run('python train.py -n 300 -r SmallNet_beta_sensitivity_'+str(beta)+' -b 128 -d 30 -v '+str(beta) + ' -l ' + str(2e-4) + ' -s cifar10 ', shell=True)
