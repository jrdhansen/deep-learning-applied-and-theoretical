'''
    File name: prob4_3_4.py
    Author: Jared Hansen
    Date created: 01/24/2019
    Date last modified: 01/24/2019
    Python Version: 3.6
'''


'''
CODE FOR HW2, PROBLEM 4.3 and 4.4 IN [DEEP LEARNING: THEORY AND APPLICATIONS]

PROMPT 4.3: For each possible input of the MLP in Figure 1 (see PDF), calculate
            the output. Ex: what is the output if X = [0,0,0], X = [1,0,0], etc
            You should have 8 cases total.

PROMPT 4.4: If we change the perceptrons in the MLP in Figure 1 (see PDF) to
            sigmoid neurons what are the outputs for the same inputs?
            (e.g. inputs of [0,0,0], [1,0,0],...)
'''





#==============================================================================
#==== Import statements =======================================================
#==============================================================================

import numpy as np
import math

# These values will be universally used, so I'm defining them at the top of
# the program. 
hl_n1_wts = np.array([0.6, 0.5, -0.6])   #==hidden layer, node1 weights
hl_n2_wts = np.array([-0.7, 0.4, 0.8])   #==hidden layer, node2 weights
ol_wts = np.ones(2)                      #==output layer weights






#==============================================================================
#==== Function definitions ====================================================
#==============================================================================

def sigmoid(z):
    """
    Returns the sigmoid function output for a given input z.
    """
    return (1 / (1 + math.exp(-z)))


def calc_node_output(wts_vec, input_vec, bias, sig_true):
    """
    This function will calculate the output value for a single perceptron node.
    First we calculate the weighted sum (wtd_sum) and then use this to
    determine the output. Output will depend on if we use the sigmoid function
    or just the typical decision rule.
    
    Parameters
    ----------
        wts_vec : numpy ndarray
            The vector of weights to calculate wtd_sum.
        input_vec : numpy ndarray
            The vector of input values to calculate wtd_sum.
        bias : int
            The bias for that particular perceptron.
        sig_true : bool
            True means use sigmoid, False means use typical decision rule.
            
    Returns
    -------
        node_val : float
            The output of the perceptron.
    """
    # Calculate the weighted sum of the inputs.
    wtd_sum = np.dot(wts_vec, input_vec) + bias
    # If we're using the sigmoid function, calculate output accordingly.
    if(sig_true):
        node_val = sigmoid(wtd_sum)        
    # If we're not using the sigmoid function, use typical decision rule.
    else:
        if(wtd_sum > 0):
            node_val = 1
        else:
            node_val = 0
    return node_val
        


def calc_final_output(input_vec, sig_true):
    """
    This function returns the final output of the pictures MLP. It rounds the
    output to 6 decimal places.
    
    Parameters
    ----------
        input_vec : numpy ndarray
            The input vector. Ex: [1,0,0], [0,1,1], etc.
        sig_true : bool
            Are we using sigmoid neurons (True) or typical perceptrons (False).
            
    Returns
    -------
        ol_output : float
            The final output of the network, rounded to 6 decimal places.
    """
    if(sig_true):
        hl_n1_out = calc_node_output(hl_n1_wts, input_vec, -0.4, True)
        hl_n2_out = calc_node_output(hl_n2_wts, input_vec, -0.5, True)
        ol_inputs = np.array([hl_n1_out, hl_n2_out])
        ol_output = calc_node_output(ol_wts, ol_inputs, -0.5, True)
    else:
        hl_n1_out = calc_node_output(hl_n1_wts, input_vec, -0.4, False)
        hl_n2_out = calc_node_output(hl_n2_wts, input_vec, -0.5, False)
        ol_inputs = np.array([hl_n1_out, hl_n2_out])
        ol_output = calc_node_output(ol_wts, ol_inputs, -0.5, False)
    print(np.round(ol_output, 6))




#==============================================================================
#==== Procedural Programming ==================================================
#==============================================================================

#=== The 8 possible different inputs we can have for the MLP.
case_1 = np.array([0,0,0])
case_2 = np.array([1,0,0])
case_3 = np.array([0,1,0])
case_4 = np.array([0,0,1])
case_5 = np.array([1,1,0])
case_6 = np.array([1,0,1])
case_7 = np.array([0,1,1])
case_8 = np.array([1,1,1])

#=== THESE ARE THE ANSWERS TO QUESTION 4.3.
#=== The MLP output for each of the 8 cases, using typical decision rule.
calc_final_output(case_1, False)
calc_final_output(case_2, False)
calc_final_output(case_3, False)
calc_final_output(case_4, False)
calc_final_output(case_5, False)
calc_final_output(case_6, False)
calc_final_output(case_7, False)
calc_final_output(case_8, False)

#=== THESE ARE THE ANSWERS TO QUESTION 4.4.
#=== The MLP output for each of the 8 cases, using typical decision rule.
calc_final_output(case_1, True)
calc_final_output(case_2, True)
calc_final_output(case_3, True)
calc_final_output(case_4, True)
calc_final_output(case_5, True)
calc_final_output(case_6, True)
calc_final_output(case_7, True)
calc_final_output(case_8, True)
