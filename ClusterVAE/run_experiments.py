import subprocess
import os
import numpy as np

# subprocess.Popen('python /home/srinath/Project/CSC413_Project/ClusterVAE/train.py -r test_automate -b 64 -d 10 -s mnist', shell=False)

# latent_dims = [5, 10, 20]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     os.system('python train.py -r latent_dem_sensitivity -b 64 -d '+str(latent_dim) + ' -s mnist ')

## MNIsT
## https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change/53046624
# batch_sizes = [32, 64, 128, 256, 1024, 4096]
# lrs         = 1e-4* np.asarray(batch_sizes, dtype= float)/batch_sizes[0]
# print(lrs)

# print(batch_sizes[0:3], lrs[0:3])
# for batch_size, lr in zip(batch_sizes[0:3], lrs[0:3]):
#     subprocess.run('python train.py -r BS_sensitivity -b '+ str(batch_size) + ' -l ' + str(lr) + ' -d 10 -s mnist', shell=True)


## CIFAR10
# batch_sizes = np.array([64, 128, 256, 1024])
# lrs         = 1e-4* np.asarray(batch_sizes, dtype= float)/batch_sizes[0]
# print(lrs)

# print(batch_sizes[[0,3]], lrs[[0,3]])
# for batch_size, lr in zip(batch_sizes[[0,3]], lrs[[0,3]]):
#     subprocess.run('python train.py -n 400 -r BS_sensitivity -b '+ str(batch_size) + ' -l ' + str(lr) + ' -d 50 -s cifar10', shell=True)


# latent_dims = [10, 20]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     subprocess.run('python train.py -n 400 -r latent_dem_sensitivity -b 64 -d '+str(latent_dim) + ' -s cifar10 ', shell=True)

# latent_dims = [10, 20, 30]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     subprocess.run('python train.py -n 400 -r latent_dem_sensitivity -b 1024 -d '+str(latent_dim) + ' -l ' + str(20e-4) + ' -s cifar10 ', shell=True)

# latent_dims = [30, 50]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     subprocess.run('python train.py -n 300 -r latent_dem_sensitivity_normalized -b 256 -d '+str(latent_dim) + ' -l ' + str(4e-4) + ' -s cifar10 ', shell=True)

# latent_dims = [30, 50]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     subprocess.run('python train.py -n 300 -r latent_dem_sensitivity_normalized_lr_e_4 -b 256 -d '+str(latent_dim) + ' -s cifar10 ', shell=True)


# latent_dims = [30, 50]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     subprocess.run('python train.py -n 300 -r SmallNet_latent_dem_sensitivity_normalized_ -b 256 -d '+str(latent_dim) + ' -l ' + str(4e-4) + ' -s cifar10 ', shell=True)

batch_sizes = [32, 64, 128]
lrs         = 1e-4* np.asarray(batch_sizes, dtype= float)/batch_sizes[0]
# latent_dims = [5]
for batch_size, lr in zip(batch_sizes, lrs):
    subprocess.run('python train.py -n 300 -r SmallNet_batch_sensitivity_normalized_ -d 30 -b '+str(batch_size) + ' -l ' + str(lr) + ' -s cifar10 ', shell=True)

