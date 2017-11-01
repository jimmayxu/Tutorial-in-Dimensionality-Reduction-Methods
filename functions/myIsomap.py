# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:17:46 2017

@author: zx621293
"""

def myIsomap (X,  color,  n_neighbors, n_components=2):
    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    t1 = time()
    fig = plt.figure(figsize=(15, 8))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("Isomap with %i neighbors considered for each data point (%.2g sec)" % (n_neighbors, t1 - t0))
    plt.axis('tight')
    plt.xlabel("first principal component",fontsize=14)
    plt.ylabel("second principal component",fontsize=14)
    plt.show()
    return;