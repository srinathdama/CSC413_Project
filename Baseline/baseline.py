import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture 
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

import seaborn as sns; sns.set()
import pandas as pd
# import os

import utils 


def map_cluster_to_labels(clusters, labels_t, noof_clusters):
    labels = np.zeros_like(clusters)
    for i in range(noof_clusters):
        mask = (clusters == i)
        labels[mask] = mode(labels_t[mask])[0]
    return labels

def plot_confusion_matrix(target, labels, target_names, figname='confusion_matrix.png'):
    mat = confusion_matrix(target, labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(figname)
    plt.clf()

def run_clustering(method, noof_clusters, target_names, mnist_path, data_set):

    noof_clusters = noof_clusters
    target_names  = target_names
    mnist_path    = mnist_path
    data_set      = data_set
    method        = method

    # load train dataset
    if data_set == 'MNIST':
        [images_tr, labels_tr] = utils.load_mnist(mnist_path)
    elif data_set == 'Fashion-MNIST':
        [images_tr, labels_tr] = utils.load_mnist_gz(mnist_path)
       
    # kmeans
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    if method   == 'Kmeans':
        model   = KMeans(n_clusters=noof_clusters, random_state=0)
    elif method == 'GMM':
        model   = GaussianMixture(n_components=noof_clusters, covariance_type='full', random_state=42)
        
    clusters = model.fit_predict(images_tr.reshape(images_tr.shape[0], -1))

    if method == 'Kmeans':
        fig, ax = plt.subplots(2, 5, figsize=(8, 3))
        centers = model.cluster_centers_.reshape(noof_clusters, images_tr.shape[1], images_tr.shape[1])
        for axi, center in zip(ax.flat, centers):
            axi.set(xticks=[], yticks=[])
            axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
        plt.savefig(F'{data_set}_kmeans_centers.png')
        plt.clf()
    elif method == 'GMM':
        print(model.converged_)
        if not model.converged_:
            print('GMM not converged')
            return

    labels    = map_cluster_to_labels(clusters, labels_tr, noof_clusters)
    train_acc = accuracy_score(labels_tr, labels)
    train_ARI = adjusted_rand_score(labels_tr, labels)
    train_NMI = normalized_mutual_info_score(labels_tr, labels)
    print('train_acc :', train_acc)
    plot_confusion_matrix(labels_tr, labels, target_names, 
                    F'{data_set}_{method}_'+str(noof_clusters)+'_train_confusion_matrix.png')


    # accuracy on test dataset
    # load test dataset
    if data_set == 'MNIST':
        [images_te, labels_te] = utils.load_mnist(mnist_path, 't10k')
    elif data_set == 'Fashion-MNIST':
        [images_te, labels_te] = utils.load_mnist_gz(mnist_path, 't10k')

    clusters  = model.predict(images_te.reshape(images_te.shape[0], -1))
    labels    = map_cluster_to_labels(clusters, labels_te, noof_clusters)
    test_acc  = accuracy_score(labels_te, labels)
    test_ARI = adjusted_rand_score(labels_te, labels)
    test_NMI = normalized_mutual_info_score(labels_te, labels)
    print('test_acc :', test_acc)
    plot_confusion_matrix(labels_te, labels, target_names, 
                    F'{data_set}_{method}_'+str(noof_clusters)+'_test_confusion_matrix.png')

    mnist_results = pd.DataFrame({
                                        'noof_clusters':[noof_clusters],
                                        'train accuracy':[train_acc],
                                        'test accuracy':[test_acc],
                                        'train ARI':[train_ARI],
                                        'test ARI':[test_ARI],
                                        'train NMI':[train_NMI],
                                        'test NMI':[test_NMI],
                                        })
    mnist_results.to_csv(F'{data_set}_{method}_'+str(noof_clusters)+'_results.csv')

    return None


if __name__ == '__main__':
    
    noof_clusters = 10
    target_names  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    mnist_path    = '/home/srinath/Project/CSC413_Project/DATASETS/MNIST/raw/'
    data_set      = 'MNIST'

    run_clustering('Kmeans', noof_clusters, target_names, mnist_path, data_set)

    run_clustering('GMM', noof_clusters, target_names, mnist_path, data_set)

    fmnist_path   = '/home/srinath/Project/CSC413_Project/DATASETS/Fashion-MNIST/raw/'
    data_set      = 'Fashion-MNIST'

    run_clustering('Kmeans', noof_clusters, target_names, fmnist_path, data_set)

    run_clustering('GMM', noof_clusters, target_names, fmnist_path, data_set)

