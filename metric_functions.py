'''
	This file contains functions for printing performance metrics for the IDA-2016 challenge.
	Function-1 ===> Misclassification cost.
	Function-2 ===> Confusion, precision and recall matrices.
'''

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

############# Misclassification cost ###################
def misclassification_score(y_true, y_pred):
    '''
        This function is used to calculate the cost metric for each model.
    '''
    # calculating confusion metrics
    conf_ = confusion_matrix(y_true, y_pred)
    fp = conf_[0,1]
    fn = conf_[1,0]
    return (10*fp) + (500*fn)


############ Confusion, Precision and Recall Matrix ##############
def plot_matrices(y_true, y_pred):
    '''
        This function is going to plot the confusion, precision and recall metrics for the model predictions.
    '''
    # confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    # precision matrix - column sum of confusion matrix
    precision = confusion/confusion.sum(axis=0)
    # recall matrix - row sum of confusion matrix
    recall = (confusion.T/confusion.sum(axis=1)).T
    
    # plot these matrices
    fig, ax = plt.subplots(ncols=3, figsize=(15,6))
    sns.heatmap(confusion, cbar=False, annot=True, ax=ax[0], fmt='g', cmap='YlGnBu')
    sns.heatmap(precision, cbar=False, annot=True, ax=ax[1], fmt='g', cmap='YlGnBu')
    sns.heatmap(recall, cbar=False, annot=True, ax=ax[2], fmt='g', cmap='YlGnBu')
    
    ax[0].set_title('Confusion matrix')
    ax[1].set_title('Precision matrix')
    ax[2].set_title('Recall matrix')
    
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")
    ax[2].set_xlabel("Predicted")
    ax[2].set_ylabel("Actual")
    plt.show()