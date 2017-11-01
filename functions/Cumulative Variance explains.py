# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:50:35 2017

@author: zx621293
"""

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
plt.show()}
