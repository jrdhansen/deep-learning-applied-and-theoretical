'''
    File name: hw2_prob2_2.py
    Author: Jared Hansen
    Date created: 01/28/2019
    Date last modified: 02/03/2019
    Python Version: 3.6
'''


'''
CODE FOR HW2, PROLEM 2.2 IN [DEEP LEARNING: THEORY AND APPLICATIONS]

PROMPT: Write a program that applies a k-nn classifier to the data with
        k in {1,5,10,15}.
        -- Calculate the test error using both leave-one-out cross-validation
           (LOOCV) and 5-fold CV.
        -- Plot the test error as a function of k.
        -- Do any values of k results in underfitting or overfitting?
        You may use existing methods in scikit-learn or other libraries for
        finding the k-nearest neighbors, but do not use any built-in k-nn
        classifiers. Also, do not use existing libraries or other methods for
        cross-validation.     
'''





#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import csv
import math
import random
import os
import pandas as pd
import numpy as np
import scipy.spatial.distance as sspd
import scipy.stats as spst
import matplotlib.pyplot as plt





#==============================================================================
#==== FUNCTIONS ===============================================================
#==============================================================================

def create_folds(all_data, num_folds):
    """
    This function takes in the whole data set and breaks into chunks for CV.
    
    Parameters
    ----------
    all_data : list of lists
    num_folds : int
    
    Returns
    -------
    all_folds : lists of lists
        For example, if num_folds is 5, all_folds is a list of 5 lists, with 
        each of the 5 lists having 42 lists (data points) of 8 elements.
    """
    random.shuffle(all_data)
    chunk_size = math.floor(len(all_data)/num_folds)
    all_folds = []
    for i in range(num_folds):
        all_folds.append(all_data[i*chunk_size:(i+1)*chunk_size])
    all_folds[-1] += all_data[num_folds*chunk_size:]
    return all_folds


def gen_pred(x, train_data, k):
    """
    This function generates a prediction for an individual point based on the
    majority vote of that point's k nearest neighbors in the training data.
    
    Parameters
    ----------
    x : int
        The individual data point for which a prediction is being generated.
    train_data : list of lists
        The data for which we are calculating the knn for x.
    k : int
        The number of nearest neighbors in train_data used to generate
        the prediction for x.
        
    Returns
    -------
    y_hat : int
        The most common reponse value in the knn (in train_data) for point x.
    """
    y_hat = 0
    dists = [0 for i in range(len(train_data))]   # list for calculated distances
    # Calculate the distance of the point to each row in the training data.
    for row in range(len(train_data)):
        arr_x = np.array(x[0:7])
        arr_train = np.array(train_data[row][0:7])
        dists[row] = sspd.euclidean(arr_x, arr_train)
    train_data = np.array(train_data)
    dists = np.array(dists).reshape(len(dists),1)
    # Add the distances to the training data
    train_data = np.concatenate((train_data, dists), axis=1)
    # Sort the training data by distances (lowest to highest)
    train_data = pd.DataFrame(train_data)
    train_data = train_data.sort_values(by=8, axis=0)
    train_data[9] = [i for i in range(len(train_data))]
    # Find the votes of the k-nearest neighbors, take the most common one as 
    # the prediction for this point.
    knn_resps = np.array(train_data[7][0:k])
    y_hat = spst.mode(knn_resps)[0][0]
    return y_hat


def knn_accuracy(train_data, test_data, k):
    """
    This function calculates and returns the accuracy of knn classification for
    a given chunk of test data and the data used to train it.
    
    Parameters
    ----------
    train_data : list of lists
        Data points used for determined CV test error.
    test_data : list of lists
        Data used to find the knn (and generate predicted response) for each
        point in train_data.
    k : int
        The number of nearest neighbors used to generate prediction for each pt
        
    Returns
    -------
    accuracy/len(test_data) : float
        The test accuracy for the given chunk of test_data and train_data.
    """
    accuracy = 0
    for data_point in test_data:
        x = [int(i) for i in data_point[0:-1]]
        y = int(data_point[-1])
        y_hat = gen_pred(data_point, train_data, k)  
        if y_hat == y:
            accuracy += 1
    return accuracy/len(test_data)


def cv_accuracy(all_folds, num_folds, k):
    """
    This function calculates overall CV accuracy for knn on the whole dataset.
    
    Parameters
    ----------
    all_folds : list of list of lists
        Contains a list of training data (list of lists) and test data (also a
        list of lists).
    num_folds : int
        How many folds for CV.
    k : int
        The number of nearest neighbors used to generate predictions.
        
    Returns
    -------
    cv_test_error : float
        The CV test error for the given inputs.    
    """
    accuracies = []
    for fold in all_folds:
        test_data = fold
        train_data = all_folds[:]  # deep copy (otherwise the original list is mutated)
        train_data.remove(test_data)
        new_data = []
        [new_data.extend(chunk) for chunk in train_data]
        train_data = new_data
        accuracy = knn_accuracy(train_data, test_data, k)
        accuracies.append(accuracy)
    cv_test_error = 1 - sum(accuracies)/num_folds
    return cv_test_error	





#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Set seed for reproducibility
random.seed(23456)

# Specify working directory and the file path of the data.
# CHANGE THIS BEFORE RUNNING CODE IN ORDER FOR IT TO WORK (I've zipped the CSV
# with my code.)
seed_loc = "C:/Users/jrdha/OneDrive/Desktop/_USU_Sp2019/_Moon_DeepLearning/hw2/"
seed_name = "data_seed.csv"
seed_path = seed_loc + seed_name
os.chdir(seed_loc)

# Initially read in the data.
with open('data_seed.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	all_data = list(reader)
# The data are read in as strings, convert them to floats.
new_data = []
for row in all_data:
    new_data.append([float(element) for element in row])
all_data = new_data

# Create array that has values of k specified in problem: 1,5,10,15.
k_vals = [1,5,10,15]

# Generate folds for 5-fold CV.
num_folds = 5
all_folds = create_folds(all_data, num_folds)
# Get 5-fold CV accuracies for each value of k.
five_fold_accs = []
for i in range(len(k_vals)):
    five_fold_accs.append(cv_accuracy(all_folds, num_folds, k_vals[i]))

# Generate folds for LOOCV
num_folds = len(all_data) - 1
all_folds = create_folds(all_data, num_folds)
# Get LOOCV accuracies for each value of k.
loocv_accs = []
for i in range(len(k_vals)):
    loocv_accs.append(cv_accuracy(all_folds, num_folds, k_vals[i]))

# Plot the test error as a function of k.
plt.plot(k_vals, five_fold_accs, color = "blue", linewidth = 2,
         label = "5-fold CV accuracy")
plt.plot(k_vals, loocv_accs, color = "red", linewidth = 2,
         label = "LOOCV accuracy")
plt.axis([0,16, 0,0.13])
plt.legend()
plt.ylabel("CV error")
plt.xlabel("$k \in \{1,5,10,15\}$: number of nearest neighbors used to generated predictions")
for i in range(len(k_vals)):
    plt.axvline(x = k_vals[i], color = "green", linestyle = ":")
plt.title("CV Test Errors for 5-fold and LOOCV with $k \in \{1,5,10,15\}$")
plt.show()