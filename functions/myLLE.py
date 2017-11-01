# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:15:22 2017

@author: zx621293
"""
def myLLE(X,  color,  n_neighbors, n_components=2):
    t0 = time()
    lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method='standard')
    Y = lle.fit_transform(X)
    t1 = time()
    fig = plt.figure(figsize=(15, 8))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    
    plt.title("Standard LLE with %i neighbors considered for each data point (%.2g sec)" % (n_neighbors, t1 - t0))
    plt.axis('tight')
    plt.show()
    
    lle.nbrs_
    return;
