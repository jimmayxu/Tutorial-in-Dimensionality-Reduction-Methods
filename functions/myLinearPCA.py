# -*- coding: utf-8 -*-
"""


@author: zx621293
"""

def myLinearPCA(X,label_refine,label):
    n = X.shape[0]
    if len(label_refine) != n:
        label_refine = [0]*n
        label = ['no ground truth']
        print('No ground truth provided in this dataset')   
    pca = PCA(svd_solver='randomized')
    t0 = time()
    X_pca = pca.fit_transform(X)
    t1 = time()    
    k = len(label)

    loadings = pca.components_
    

    plt.figure(figsize=(20, 10))
    plt.suptitle("Linear PCA on dataset with accepted %i experiments, each with %i covariates. \nClasses: %s. Time: %.2fs" 
                 % (X.shape[0],X.shape[1],label,t1-t0), fontsize=24)
    for i in [1,2]:
        plt.subplot(1, 2, i)
        #, aspect='equal'
        for j,lab in zip(np.linspace(0,k-1,k).astype('int'),label):
            plt.scatter(X_pca[label_refine==j, np.mod(i,2)], 
                              X_pca[label_refine==j, np.mod(i,2)+1],cmap=plt.cm.Spectral,label=lab)
        plt.xlabel("%i principal component"%(np.mod(i,2)+1),fontsize=14)
        plt.ylabel("%i principal component"%(np.mod(i,2)+2),fontsize=14)
        plt.legend(loc=4)
        plt.axis()   
    plt.show()
    
    return X_pca,loadings;

