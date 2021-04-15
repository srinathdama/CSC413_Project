import subprocess
import os

subprocess.run('python train.py -r test_automate -n 2 -b 64 -d 10 -s mnist', shell=True)

# latent_dims = [30, 50]
# # latent_dims = [5]
# for latent_dim in latent_dims:
#     os.system('python train.py -r latent_dem_sensitivity -b 64 -d '+str(latent_dim) + ' -s mnist')  