# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:40:18 2017

@author: zx621293
"""


########################using t-SNE##################################
def myTSNE(X,label_refine,label,perplexity):
    n = X.shape[0]
    if len(label_refine) != n:
        label_refine = [0]*n
        label = ['no ground truth']
        print('No ground truth provided in this dataset')
    (n_sample,n_protein) = X.shape
    l = len(perplexity)
    k = len(label)
    
    fig = plt.figure(figsize=(30,(5*l))) 
    plt.suptitle("t-SNE with accepted %i experiments, each with %i covariates. \nClasses: %s " 
                 % (X.shape[0],X.shape[1],label), fontsize=24)
    number = np.linspace(1,l,l) 
    
    YY = list()
    
    for i,perp in zip(number,perplexity):
        t0 = time()
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perp)
        Y = tsne.fit_transform(X)
        YY.append(Y)
        t1 = time()
        print("t-SNE with perpexity %2.f: %.2g sec" % (perp,t1 - t0))
        fig.add_subplot(l/2,2,i)
        for j,lab in zip(np.linspace(0,k-1,k),label):
            plt.scatter(Y[label_refine==j, 0], Y[label_refine==j, 1],cmap=plt.cm.Spectral,label=lab)
        plt.title("t-SNE with perpexity %.2f (%.2g sec)" % (perp,t1 - t0))
        plt.legend(loc=4)
        plt.axis()
    plt.show()  
    return YY; #principal component matrix