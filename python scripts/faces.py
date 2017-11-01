# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:43:17 2017

@author: zx621293
"""
exec(open("./functions.py").read())

import logging
from time import time
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 8,8
n_components = n_row*n_col
image_shape = (64, 64)
rng = RandomState(0)

#load face data


dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)



################################################################################

X = faces_centered
myTSNE(X,[0],[0],[5,10,20,50])
myLinearPCA(X,[],[])

##finding components of dimensionality reduction methods
estimator = decomposition.PCA(n_components = 6,svd_solver='randomized',whiten=True)
y = estimator.fit_transform(X)
components = estimator.components_

X_pca,com_pca = myLinearPCA(X,[],[])
X_fa,com_fa = myFA(X,[] , [],6)


plot_gallery('Eigenfaces - PCA using randomized SVD',com_pca,4,8)
plot_gallery('Factor Analysis components - FA',com_fa,3,2)

