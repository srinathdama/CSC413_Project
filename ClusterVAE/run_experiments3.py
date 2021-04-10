import subprocess
import os

path  = '/home/srinath/Project/CSC413_Project/ClusterVAE/runs/cifar10'
dir_names = [os.path.join(path, fn) for fn in next(os.walk(path))[1]]

for dir_name in dir_names:
    print(dir_name)
    subprocess.run('python criteria.py -r '+ str(dir_name), shell=True)
    subprocess.run('python gen-examples.py -r '+ str(dir_name), shell=True)
    subprocess.run('python tsne-cluster.py -r '+ str(dir_name), shell=True)
