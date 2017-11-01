# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:41:11 2017

@author: zx621293
"""



####################KERNEL PCA
def myGKernelPCA (X, label_refine,label, gamma):
    n = X.shape[0]
    if len(label_refine) != n:
        label_refine = [0]*n
        label = ['no ground truth']
        print('No ground truth provided in this dataset')
        
    gamma=gamma*2
    gamma.sort()
    l = len(gamma)
    k = len(label)
    plt.figure(figsize=(30,(5*l))) 
    plt.suptitle("Kernel PCA on dataset with accepted %i experiments, each with %i covariates. \nClasses: %s " 
                 % (X.shape[0],X.shape[1],label), fontsize=24)
    for i,gam in zip(np.linspace(1,l,l).astype("int"),gamma):
        t0=time()
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=gam)
        X_kpca = kpca.fit_transform(X)
        t1=time()
        print("Gaussian PCA with gamma %.2f: %.2g sec" % (gam,t1 - t0))
        #X_back = kpca.inverse_transform(X_kpca)        
        plt.subplot((l/2), 2, i)
        plt.title("Projection by Gaussian-kPCA, gamma = %.2f (%.2g sec)" %(gam,t1-t0))
        for j,lab in zip(np.linspace(0,k-1,k),label):
            plt.scatter(X_kpca[label_refine==j, np.mod(i,2)], X_kpca[label_refine==j, np.mod(i,2)+1]#
                               ,cmap=plt.cm.Spectral,label=lab)
        plt.xlabel("%i principal component"%(np.mod(i,2)+1),fontsize=14)
        plt.ylabel("%i principal component"%(np.mod(i,2)+2),fontsize=14)
        plt.legend(loc=1)
        plt.axis()       
    loadings = kpca.alphas_
    plt.show()
    return X_kpca,loadings;