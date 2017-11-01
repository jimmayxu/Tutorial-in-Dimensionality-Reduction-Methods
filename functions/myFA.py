# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:44:28 2017

@author: zx621293
"""

def myFA (X, label_refine, label, n_components, max_iter=2):
    n = X.shape[0]
    if len(label_refine) != n:
        label_refine = [0]*n
        label = ['no ground truth']
        print('No ground truth provided in this dataset')
        
    estimator = decomposition.FactorAnalysis(n_components=n_components, max_iter=2)
    t0=time()  
    X_fa = estimator.fit_transform(X)
    t1=time()
    plt.figure(figsize=(30,10)) 
    plt.suptitle("Factor Analysis on dataset with accepted %i experiments, each with %i covariates. \nClasses: %s " 
                 % (X.shape[0],X.shape[1],label), fontsize=24)
    
    
    k = len(label)

    for i in [1,2]:
        plt.subplot(1, 2, i)
        plt.title("Independent components - FastICA' (%.2g sec)" %(t1-t0))
        for j,lab in zip(np.linspace(0,k-1,k),label):
            plt.scatter(X_fa[label_refine==j, np.mod(i,2)], X_fa[label_refine==j, np.mod(i,2)+1]#
                               ,cmap=plt.cm.Spectral,label=lab)
        plt.xlabel("%i principal component"%(np.mod(i,2)+1),fontsize=14)
        plt.ylabel("%i principal component"%(np.mod(i,2)+2),fontsize=14)
        plt.legend(loc=1)
        plt.axis()
        
    plt.show()
    components = estimator.components_
    
        
    return X_fa,components;