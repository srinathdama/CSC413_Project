from __future__ import print_function

try:
    import os
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

    import matplotlib
    # matplotlib.use('Agg')
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

except ImportError as e:
    print(e)
    raise ImportError



# Nan-avoiding logarithm
def tlog(x):
      return torch.log(x + 1e-8)


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


# Cross Entropy loss with two vector inputs
def cross_entropy(pred, soft_targets):
    log_softmax_pred = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.mean( torch.sum(- soft_targets * log_softmax_pred, 1) )


# Save a provided model to file
def save_model(models=[], out_dir=''):

    # Ensure at least one model to save
    assert len(models) > 0, "Must have at least one model to save."

    # Save models to directory out_dir
    for model in models:
        filename = model.name + '.pth.tar'
        outfile = os.path.join(out_dir, filename)
        torch.save(model.state_dict(), outfile)


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):

    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c) ), "Requested class %i outside bounds."%fix_class

    Tensor = torch.cuda.FloatTensor
    
    # Sample noise as generator input, zn
    zn = Variable(Tensor(0.75*np.random.normal(0, 1, (shape, latent_dim))), requires_grad=req_grad)

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = Tensor(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).cuda()
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
        #zc_idx = torch.empty(shape, dtype=torch.long).random_(n_c).cuda()
        #zc_FT = Tensor(shape, n_c).fill_(0).scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda()
        zc_FT = zc_FT.cuda()

    zc = Variable(zc_FT, requires_grad=req_grad)

    ## Gaussian-noisey vector generation
    #zc = Variable(Tensor(np.random.normal(0, 1, (shape, n_c))), requires_grad=req_grad)
    #zc = softmax(zc)
    #zc_idx = torch.argmax(zc, dim=1)

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


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

def run_clustering(method, noof_clusters, target_names, save_path, data_set, train_data, test_data):

    noof_clusters = noof_clusters
    target_names  = target_names
    data_set      = data_set
    method        = method

    # load train dataset
    [images_tr, labels_tr] = train_data
       
    # kmeans
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    if method   == 'Kmeans':
        model   = KMeans(n_clusters=noof_clusters, random_state=0)
    elif method == 'GMM':
        model   = GaussianMixture(n_components=noof_clusters, covariance_type='full', random_state=42)
        
    clusters = model.fit_predict(images_tr.reshape(images_tr.shape[0], -1))


    if method == 'GMM':
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
                    os.path.join(save_path, 
                    F'{data_set}_{method}_'+str(noof_clusters)+'_train_confusion_matrix.png'))


    # accuracy on test dataset
    # load test dataset
    [images_te, labels_te] = test_data

    clusters  = model.predict(images_te.reshape(images_te.shape[0], -1))
    labels    = map_cluster_to_labels(clusters, labels_te, noof_clusters)
    test_acc  = accuracy_score(labels_te, labels)
    test_ARI = adjusted_rand_score(labels_te, labels)
    test_NMI = normalized_mutual_info_score(labels_te, labels)
    print('test_acc :', test_acc)
    plot_confusion_matrix(labels_te, labels, target_names, 
                    os.path.join(save_path, 
                    F'{data_set}_{method}_'+str(noof_clusters)+'_test_confusion_matrix.png'))

    mnist_results = pd.DataFrame({
                                        'noof_clusters':[noof_clusters],
                                        'train accuracy':[train_acc],
                                        'test accuracy':[test_acc],
                                        'train ARI':[train_ARI],
                                        'test ARI':[test_ARI],
                                        'train NMI':[train_NMI],
                                        'test NMI':[test_NMI],
                                        })
    mnist_results.to_csv(os.path.join(save_path,
                    F'{data_set}_{method}_'+str(noof_clusters)+'_results.csv'))

    return None
