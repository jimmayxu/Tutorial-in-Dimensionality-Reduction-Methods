# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:47:05 2017

@author: zx621293
"""

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