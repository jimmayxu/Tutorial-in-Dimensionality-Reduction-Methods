# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:38:54 2017

@author: zx621293
"""

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
