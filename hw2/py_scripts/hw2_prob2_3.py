'''
    File name: hw2_prob2_3.py
    Author: Jared Hansen
    Date created: 01/28/2019
    Date last modified: 02/03/2019
    Python Version: 3.6
'''

'''
CODE FOR HW2, PROLEM 2.3 IN [DEEP LEARNING: THEORY AND APPLICATIONS]

PROMPT: apply two other classifiers (other than knn) to the same data. Possible
        algorithms include (but are not limited to) logistic regression, QDA,
        naive Bayes, SVM, and decision trees. You may use existing libraries.
        Use 5-fold cross-validation to calculate the test error.
        -- Report the training and test errors.
        -- If any tuning parameters need to be selected, use cross-validation
           and report the training and test error for several values of the
           tuning parameters.
'''





#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression





#==============================================================================
#==== FUNCTIONS ===============================================================
#==============================================================================

def gen_errors(clf, method):
    """
    This function prints the 5-fold CV test error and the training error for a
    given classifier (clf).
    NOTE: I am understanding "training error" to be calculated without use of
          CV for error calculation. I train the model on the entire dataset,
          and then predict back onto the entire dataset to generate training 
          error.
    NOTE: per https://scikit-learn.org/stable/modules/multiclass.html,
          "All classifiers in scikit-learn do multiclass classification
          out-of-the-box." Hence, there's no need to do anything like
          "one-V-all" classification to use logistic regression like we might
          in another language (since it's a binary method.)
          
    Parameters
    ----------
    clf: method
        This will be either RandomForestClassifier() or LogisticRegresion().
        The classifier used to generate the error rates.
    method : string
        Used for automatic printing of labeled error rates to the console.
        
    Returns
    -------
    Nothing (prints error rates)    
    """
    # Set the classifier to be the classifier input to the function.
    clf = clf
    # 5-Fold CV error for classifier.
    print("The 5-fold CV test error for", method, "is: ",
          1- np.mean(cross_val_score(clf, features, labels, cv=5)))
    # Train the classifier on all of the data.
    clf.fit(features, labels)
    # Generate classifier predictions for training error (on all data).
    train_preds = clf.predict(features)
    # Create a vector comparing the predictions to the true response values:
    # a value of 0 ==> correct pred, any other number ==> incorrect pred.
    train_correct = train_preds - seed_data.resp
    # Count the number of non-zero values and divide by length of data to
    # obtain training error.
    train_error = (np.count_nonzero(train_correct))/(len(train_correct))
    print("The training error for", method, "is: ", train_error)
    
    
    
    
    
#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Set the path of the data.
# THIS WILL NEED TO BE CHANGED IF RUNNING ON ANOTHER MACHINE.
seed_loc = "C:/Users/jrdha/OneDrive/Desktop/_USU_Sp2019/_Moon_DeepLearning/hw2/"
seed_name = "data_seed.csv"
seed_path = seed_loc + seed_name

# Import the data as a pandas dataframe.
col_names = ["pred1","pred2","pred3","pred4","pred5","pred6","pred7","resp"]
seed_data = pd.read_csv(seed_path, header=None, names=col_names) 

# Set a seed for reproducibility.
np.random.seed(2345)

#=== Data Pre-processing ======================================================
# Our labels are the column named "resp" in the seed_data dataframe.
labels = np.array(seed_data.resp)
# Remove the labels from the dataframe. Axis 1 refers to the columns.
features = seed_data.drop("resp", axis = 1)
features = np.array(features)

# Calculate and display the 5-fold CV test error and the training error for my
# chosen classifiers: random forests and logistic regression.
# (See note in gen_errors function about why logistic regression can be used
# for multiclass classification without any modifications.)
gen_errors(RandomForestClassifier(), "random forests")
gen_errors(LogisticRegression(), "logistic regression")