
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:41:17 2017

@author: zx621293.

FINISH LOADING
"""

print(__doc__)


from time import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn import manifold


from sklearn import decomposition

Axes3D




####################refine data
def RefineData (protein,labelling): #(input data, labelled ground truth)
    mydata=protein.values.astype("float64")
    delrow= list()
    delcolumn= list()
    for column in np.linspace(0,mydata.shape[1]-1,mydata.shape[1]).astype('int'):
        if sum(mydata[:,column]!=mydata[:,column])>3: #number of nan in each row
            delcolumn.append(column)
    print('%i proteins covariates are deleted due to more than three missing value for that protein' %len(delcolumn))
    print(delcolumn)
    
    
    for row in np.linspace(0,mydata.shape[0]-1,mydata.shape[0]).astype('int'):
        if sum(mydata[row]!=mydata[row])>=3: #number of nan in each row
            delrow.append(row)
    print('%i experiments are deleted due to more than three missing protein value for that experiment' %len(delrow))
    print(delrow)
            
    mydata_refine = np.delete(mydata,(delrow),0)
    mydata_refine = np.delete(mydata_refine,(delcolumn),1)
    label_each = np.array([labelling])
    label_refine = np.delete(label_each,(delrow),1)
    
    print('Check no more missing value in our refined dataset: %s' %((mydata_refine!=mydata_refine).sum()==0))
    
    if (mydata_refine!=mydata_refine).sum()>0:
        mydata = mydata[np.all(mydata > 0, axis=1)]
        label_refine = label_refine[np.all(mydata > 0, axis=1)]

            
    print('The resulting input data matrix is refined from \n%i by %i \nto \n%i by %i dimension' %(protein.shape+mydata_refine.shape), )

    return mydata_refine, label_refine[0], delrow, delcolumn; #(refined input data with refined labelled ground truth)














#using dimensionality reduction methods

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
    plt.suptitle("t-SNE on mice data with accepted %i mice experiments, each with %i covariates. \nClasses: %s " 
                 % (X.shape[0],X.shape[1],label), fontsize=24)
    

    number = np.linspace(1,l,l)
    
    
    for i,perp in zip(number,perplexity):
        t0 = time()
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0, perplexity=perp)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("t-SNE with perpexity %2.f: %.2g sec" % (perp,t1 - t0))
        fig.add_subplot(l/2,2,i)
        for j,lab in zip(np.linspace(0,k-1,k),label):
            plt.scatter(Y[label_refine==j, 0], Y[label_refine==j, 1],cmap=plt.cm.Spectral,label=lab)
        plt.title("t-SNE with perpexity %.2f (%.2g sec)" % (perp,t1 - t0))
        plt.legend(loc=4)
        plt.axis()
    
    plt.show()  
    
    return Y; #principal component matrix



##################LINEAR PCA

def myLinearPCA(X,label_refine,label):
    n = X.shape[0]
    if len(label_refine) != n:
        label_refine = [0]*n
        label = ['no ground truth']
        print('No ground truth provided in this dataset')   
    pca = PCA(svd_solver='randomized')
    X_pca = pca.fit_transform(X)
    k = len(label)

    loadings = pca.components_
    

    plt.figure(figsize=(20, 10))
    plt.suptitle("Linear PCA on dataset with accepted %i experiments, each with %i covariates. \nClasses: %s " 
                 % (X.shape[0],X.shape[1],label), fontsize=24)
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
    
    
    

    #Cumulative Variance explains
    var= pca.explained_variance_ratio_
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle('Cumulative Variance contribution for linear PCA on mice data.',fontsize=24)
    fig.add_subplot(1,2,1)
    plt.scatter(np.linspace(1,len(var),len(var)),var,color='red')
    plt.xlabel('#th principal components',fontsize=14)
    plt.title('Proportion of covariance explained',fontsize=14)
    fig.add_subplot(1,2,2)
    plt.plot(var1)
    plt.title('Accumulative proportion of covariance explained',fontsize=14)
    plt.xlabel('first # of principal components',fontsize=14)
    plt.ylabel('percentage (%)',fontsize=14)
    plt.axis()
    
    return X_pca,loadings;

#The amount of variance that each PC explains













###



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
    plt.suptitle("Kernel PCA on dataset with accepted %i mice experiments, each with %i covariates. \nClasses: %s " 
                 % (X.shape[0],X.shape[1],label), fontsize=24)
    for i,gam in zip(np.linspace(1,l,l).astype("int"),gamma):
        t0=time()
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=gam)
        X_kpca = kpca.fit_transform(X)
        t1=time()
        print("Gaussian PCA with gamma %2.f: %.2g sec" % (gam,t1 - t0))
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
#

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

def Covariates_interpretation (components,delcolumn,covariate_list):
    min_max_scaler = preprocessing.MinMaxScaler()
    p = min_max_scaler.fit_transform(components.T).T
    p1 = p[:,0]
    p2 = min_max_scaler.fit_transform(p[:,(0,1)].sum(axis=1))
    p3 = min_max_scaler.fit_transform(p[:,(0,1,2)].sum(axis=1))
    
    xaxis = np.linspace(1,len(p3),len(p3))
    
    plt.figure(figsize=(20, 10)) 
    plt.plot(xaxis,p3,'--ro',label = 'first three principal component')
    plt.plot(xaxis,p2,'--bo',label = 'first two principal component')
    plt.plot(xaxis,p1,'--go',label='first principal component')
    plt.title("The contribution of each protein to first few principal components",
              fontsize=20)
    plt.xlabel('Protein covariate')
    plt.ylabel('Proportion contributed to leading principal component')
    plt.legend(loc=1,fontsize=15)
    
    #make a table for the three most influencing covariates
    ppmax = np.zeros([3,3]).astype('int')
    for i,pp in zip([0,1,2],[p1,p2,p3]):
        for j in [0,1,2]:
            ppmax[i,j] = np.argmax(pp)
            pp[ppmax[i,j]]=0
                
    #consider about the rearranged order of protein due to dataset refinement

    import copy
    protein_refine = copy.copy(covariate_list)
    
    for i in delcolumn:
        protein_refine.remove(covariate_list[i])     
     
    table = np.zeros([4,4]).astype('str')  
    table[0,0] = 'first # principal components'
    if delcolumn != 'NA':
        for i,col in zip([1,2,3],delcolumn):
            table[i,0] = 'first %i principal components'%(i)
            table[i,1:4] = np.array(protein_refine)[list(ppmax[i-1])]
        for j in [1,2,3]:
            table[0,j] = '%ith covariate'%j    
    #print array table
        print('\n'.join(['       '.join(['{:4}'.format(item) for item in row]) 
      for row in table]))
    else:
        print('Please determine the column index deleted in the input, if no, assign second input as: NA ')

    plt.show()
    return table;








