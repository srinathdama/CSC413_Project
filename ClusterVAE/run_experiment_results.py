import subprocess
import os

# path  = '/home/srinath/Project/CSC413_Project/ClusterVAE/runs/cifar10'
# dir_names = [os.path.join(path, fn) for fn in next(os.walk(path))[1]]

# dir_names = ['/home/srinath/Project/CSC413_Project/ClusterVAE/runs/cifar10/300epoch_z50_vae_vanilla_bs256_latent_dem_sensitivity_normalized']


# path  = '/home/srinath/Project/CSC413_Project/ClusterVAE/runs/cifar10'
# dir_names = [os.path.join(path, '300epoch_z30_vae_vanilla_bs128_SmallNet1_gaussian_beta_sensitivity_1'),
#             os.path.join(path, '300epoch_z30_vae_vanilla_bs128_SmallNet1_gaussian_beta_sensitivity_2'),
#             os.path.join(path, '300epoch_z30_vae_vanilla_bs128_SmallNet1_gaussian_beta_sensitivity_3'),
#             os.path.join(path, '300epoch_z30_vae_vanilla_bs128_SmallNet1_gaussian_beta_sensitivity_4'),
#             os.path.join(path, '300epoch_z30_vae_vanilla_bs128_SmallNet1_gaussian_beta_sensitivity_5')]

path  = '/home/srinath/Project/CSC413_Project/ClusterVAE/runs/cifar10'
dir_names = [os.path.join(path, '300epoch_z50_vae_vanilla_bs128_cifar_k3_gaussian_sigma_sensitivity_0.8')]


for dir_name in dir_names:
    print(dir_name)
    subprocess.run('python criteria.py -r '+ str(dir_name), shell=True)
    subprocess.run('python gen-examples.py -r '+ str(dir_name), shell=True)
    subprocess.run('python tsne-cluster.py -r '+ str(dir_name), shell=True)
