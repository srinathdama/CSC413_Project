import os
import gzip
import numpy as np
import struct
from datasets import get_dataloader, dataset_list

def load_mnist_gz(path, kind='train'):

    """Load MNIST data from `path`"""
    ''' https://github.com/aksharas28/Unsupervised-Learning--Clustering
    -Analysis-on-Fashion-MNIST-Data/blob/master/util_mnist_reader.py'''

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16)
        images = images.reshape(len(labels), 28, 28)

    return images, labels

def load_mnist(path, kind='train'):

    """Load MNIST data idx files from `path`"""
    '''https://stackoverflow.com/questions/62958011/
    how-to-correctly-parse-mnist-datasetidx-format-into-python-arrays '''


    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    def read_label_idx_file(file):
        with open(file,'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape(-1)
        return data
    
    def read_images_idx_file(file):
        with open(file,'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        return data

    labels = read_label_idx_file(labels_path)
    
    images = read_images_idx_file(images_path)

    return [images, labels]


def load_data(data_set, data_dir, train_flag):
    batch_size = 5000
    dataloader = get_dataloader(dataset_name=data_set,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                train_set=train_flag)

    images = []
    labels = []

    for i, (imgs, itruth_label) in enumerate(dataloader):
        images.append(imgs.numpy())
        labels.append(itruth_label.numpy())
    
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return [images, labels]