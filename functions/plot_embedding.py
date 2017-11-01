# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:10:23 2017

@author: zx621293
"""

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):

    plt.figure()#############
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(digits.target[i]),color=plt.cm.Set1(digits.target[i] ),
                 fontdict={'weight': 'bold', 'size': 9})
########################plotting numbers

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue #!!!!it applies to if loop
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)    
    plt.axis()
    plt.show()