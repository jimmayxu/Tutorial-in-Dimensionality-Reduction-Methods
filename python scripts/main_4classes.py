# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:11:35 2017

@author: zx621293
"""
exec(open("./functions.py").read())

import pandas as pd
from sklearn import preprocessing

mydata = pd.read_csv('C:/Users/zx621293/OneDrive - GSK/PROJECT/Python Scripts/MICE data/GSK-PYTHON/Data_Cortex_Nuclear.csv')
#########################################################
protein = mydata.T[:82][2:].T

protein= protein[protein["Behavior"]=="C/S"]
CSSC = protein["Genotype"]=="Ts65Dn"
labelling = CSSC.astype('int')
labelling[protein['class']=='t-CS-m']=2
protein = protein.T[:76].T



#data refinement
(mydata_refine, label_refine,delrow, delcolumn) = RefineData(protein,labelling)


label = ['Normal Learning','Failed Learning','Rescued Learning']


covariate_list = list(protein.columns.values.astype('str'))

#PREPROCESSING
X = preprocessing.scale(mydata_refine)



############################dimensionality redunction methods#####################


#using linear PCA

X_pca,components = myLinearPCA(X,label_refine,label)

table= Covariates_interpretation(components,delcolumn,covariate_list)


#using Gaussian kernel PCA

X_kpca = myGKernelPCA (X, label_refine,label, [0.1,20,100])


Covariates_interpretation(X_kpca,delcolumn,covariate_list)


#using t-SNE method
X_TSNE = myTSNE(X,label_refine,label,[5,15,30,50])

Covariates_interpretation(X_TSNE,delcolumn,covariate_list)


print('Now find the dominating protein in the first two principal component')