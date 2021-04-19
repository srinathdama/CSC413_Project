import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

import seaborn as sns; sns.set()
import pandas as pd
# import os
from sklearn.manifold import TSNE

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

def run_clustering(method, noof_clusters, target_names, data_dir,
                 data_set_name, run_dir, pca_dim = None):

    noof_clusters = noof_clusters
    target_names  = target_names
    data_dir    = data_dir
    data_set_name      = data_set_name
    method        = method

    # load train dataset
    # if data_set_name == 'mnist':
    #     [images_tr, labels_tr] = utils.load_mnist(data_dir)
    # elif data_set_name == 'fashion-mnist':
    #     [images_tr, labels_tr] = utils.load_mnist_gz(data_dir)
    # elif data_set_name == 'cifar10':
    #     [images_tr, labels_tr] = utils.load_data(data_set_name, data_dir, True)
    #     images_tr = np.mean(images_tr, axis=1)

    [images_tr, labels_tr] = utils.load_data(data_set_name, data_dir, True)
    images_tr = np.mean(images_tr, axis=1)

    # kmeans
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    if method   == 'Kmeans':
        model   = KMeans(n_clusters=noof_clusters, random_state=0)
    elif method == 'GMM':
        model   = GaussianMixture(n_components=noof_clusters, covariance_type='full', random_state=42)
        
    if pca_dim:
        pca = PCA(n_components=pca_dim, svd_solver='full', whiten=True)
        images_tr = pca.fit_transform(images_tr.reshape(images_tr.shape[0], -1))
        clusters  = model.fit_predict(images_tr)
    else:
        clusters = model.fit_predict(images_tr.reshape(images_tr.shape[0], -1))

    if (method == 'Kmeans' and not pca_dim):
        fig, ax = plt.subplots(2, 5, figsize=(8, 3))
        centers = model.cluster_centers_.reshape(noof_clusters, images_tr.shape[1], images_tr.shape[1])
        for axi, center in zip(ax.flat, centers):
            axi.set(xticks=[], yticks=[])
            axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
        plt.savefig(os.path.join(run_dir, F'{data_set_name}_kmeans_centers.png'))
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
                    os.path.join(run_dir, 
                    F'{data_set_name}_{method}_'+str(noof_clusters)+'_train_confusion_matrix.png'))


    # accuracy on test dataset
    # load test dataset

    # if data_set_name == 'mnist':
    #     [images_te, labels_te] = utils.load_mnist(data_dir, 't10k')
    # elif data_set_name == 'fashion-mnist':
    #     [images_te, labels_te] = utils.load_mnist_gz(data_dir, 't10k')
    # elif data_set_name == 'cifar10':
    #     [images_te, labels_te] = utils.load_data(data_set_name, data_dir, False)
    #     images_te = np.mean(images_te, axis=1)

    [images_te, labels_te] = utils.load_data(data_set_name, data_dir, False)
    images_te = np.mean(images_te, axis=1)

    if pca_dim:
        images_te = pca.transform(images_te.reshape(images_te.shape[0], -1))
        clusters  = model.predict(images_te)
    else:
        clusters  = model.predict(images_te.reshape(images_te.shape[0], -1))

    labels    = map_cluster_to_labels(clusters, labels_te, noof_clusters)
    test_acc  = accuracy_score(labels_te, labels)
    test_ARI = adjusted_rand_score(labels_te, labels)
    test_NMI = normalized_mutual_info_score(labels_te, labels)
    print('test_acc :', test_acc)
    plot_confusion_matrix(labels_te, labels, target_names, 
                    os.path.join(run_dir, 
                    F'{data_set_name}_{method}_'+str(noof_clusters)+'_test_confusion_matrix.png'))

    mnist_results = pd.DataFrame({
                                        'noof_clusters':[noof_clusters],
                                        'train accuracy':[train_acc],
                                        'test accuracy':[test_acc],
                                        'train ARI':[train_ARI],
                                        'test ARI':[test_ARI],
                                        'train NMI':[train_NMI],
                                        'test NMI':[test_NMI],
                                        })
    mnist_results.to_csv(os.path.join(run_dir, 
                    F'{data_set_name}_{method}_'+str(noof_clusters)+'_results.csv'))


    # Load TSNE
    # run_dir = 
    perplexity = -1
    if (perplexity < 0):
        tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
        fig_title = "PCA Initialization"
        figname = os.path.join(run_dir, F'{data_set_name}_tsne-pca.png')
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
        fig_title = "Perplexity = $%d$"%perplexity
        figname = os.path.join(run_dir, F'{data_set_name}_tsne-plex%i.png'%perplexity)

    tsne_enc = tsne.fit_transform(images_te.reshape(images_te.shape[0], -1)[0:1000])

    # Convert to numpy for indexing purposes
    labels = labels_te[0:1000]

    # Color and marker for each true class
    colors = cm.rainbow(np.linspace(0, 1, noof_clusters))
    markers = matplotlib.markers.MarkerStyle.filled_markers

    # Save TSNE figure to file
    fig, ax = plt.subplots(figsize=(16,10))
    for iclass in range(0, noof_clusters):
        # Get indices for each class
        idxs = labels==iclass
        # Scatter those points in tsne dims
        ax.scatter(tsne_enc[idxs, 0],
                   tsne_enc[idxs, 1],
                   marker=markers[iclass],
                   c=colors[iclass],
                   edgecolor=None,
                   label=r'$%i$'%iclass)

    ax.set_title(r'%s'%fig_title, fontsize=24)
    ax.set_xlabel(r'$X^{\mathrm{tSNE}}_1$', fontsize=18)
    ax.set_ylabel(r'$X^{\mathrm{tSNE}}_2$', fontsize=18)
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('2')
    plt.legend(title=r'Class', loc='best', numpoints=1, fontsize=16)
    plt.tight_layout()
    ax = plt.gca()
    ax.set_facecolor('w')
    fig.savefig(figname)

    return None


if __name__ == '__main__':

    # Local directory of CypherCat API
    BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Local directory containing entire repo
    # REPO_DIR = os.path.split(BASELINE_DIR)[0]

    noof_clusters = 10
    target_names  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # dataset_paths    = ['/home/srinath/Project/CSC413_Project/DATASETS/MNIST/raw/',
    #                     '/home/srinath/Project/CSC413_Project/DATASETS/Fashion-MNIST/raw/',
    #                     '/home/srinath/Project/CSC413_Project/ClusterVAE/datasets/cifar10/']
    
    dataset_paths    = ['/home/srinath/Project/CSC413_Project/DATASETS/MNIST/raw/',
                        '/home/srinath/Project/CSC413_Project/ClusterVAE/datasets/fashion-mnist/',
                        '/home/srinath/Project/CSC413_Project/ClusterVAE/datasets/cifar10/']

    data_sets      = ['mnist', 'fashion-mnist', 'cifar10']
    pca_flag       = True  #None #50
    run_name       = 'pca_all' #'baseline_full_size' #'pca'


    # for data_path, data_set in zip([dataset_paths[2]], [data_sets[2]]):
    for data_path, data_set in zip(dataset_paths, data_sets):

        print(data_set)
        run_dir = os.path.join(BASELINE_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        if pca_flag:
            if data_set=='mnist':
                pca_dim = 30
            elif data_set=='fashion-mnist':
                pca_dim = 40
            else:
                pca_dim = 50
        else:
            pca_dim = None

        print('Kmeans')
        run_clustering('Kmeans', noof_clusters, target_names, data_path, data_set, run_dir, pca_dim)

        print('GMM')
        run_clustering('GMM', noof_clusters, target_names, data_path, data_set, run_dir, pca_dim)



    # run_clustering('Kmeans', noof_clusters, target_names, data_path, data_set, pca_dim)

    # run_clustering('GMM', noof_clusters, target_names, mnist_path, data_set, pca_dim)

    # fmnist_path   = '/home/srinath/Project/CSC413_Project/DATASETS/Fashion-MNIST/raw/'
    # data_set      = 'Fashion-MNIST'

    # run_clustering('Kmeans', noof_clusters, target_names, fmnist_path, data_set, pca_dim)

    # run_clustering('GMM', noof_clusters, target_names, fmnist_path, data_set, pca_dim)

