'''
    File name: hw4_prob1_3.py
    Author: Jared Hansen
    Date created: 02/28/2019
    Date last modified: 02/28/2019
    Python Version: 3.6.4

===============================================================================
CODE FOR HW4, PROLEM 1.3 IN [DEEP LEARNING: THEORY AND APPLICATIONS]
===============================================================================
'''

'''
PROMPT: Given the network in Figure 1, calculate the derivatives of the cost 
        with respect to the weights and the biases, and the backpropogation
        error equations (i.e. delta^l for each layer l) for the first iteration
        using the cross-entropy cost function.
        
        In other words, calculate:
        -----------------------------------------------------------------------
        ** THE DELTA VECTORS:               delta_hid, delta_out
        ** PARTIAL OF COST WRT EACH WEIGHT: partial_w1,...,partial_w8            
        ** PARTIAL OF  COST WRT EACH BIAS:  partial_b_hid_1,...,partial_b_out_2
            
This script is used for performing the necessary calculations for problem 1.3.
'''






#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import numpy as np




#==============================================================================
#==== FUNCTIONS & CLASSES =====================================================
#==============================================================================

# This returns the output of the sigmoid funcion for a given input z.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# This returns the output of the sigmoid' function for a given input z.
# (sigmoid' is just the first derivative of the sigmoid  wrt z)
def sig_prime(z):
    return np.exp(-z) / ((1 + np.exp(-z))**2)
    




#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================


#=== FEED FORWARD =============================================================

# The weights matrix connecting the input layer to the hidden layer.
wts_1 = np.matrix([[0.15, 0.25],[0.20, 0.30]])
# The vector of inputs.
inputs = np.array([0.05, 0.1])
# The biases for the first hidden layer
biases_hid = np.array([0.35, 0.35])
# Compute the intermediate value "z" for the hidden layer
z_hid = np.dot(wts_1, inputs) + biases_hid
# Compute the activations for the hidden layer.
actvn_hid = sigmoid(z_hid).reshape(2,1)

# The weights matrix connecting the hidden layer to the output layer.
wts_2 = np.matrix([[0.40, 0.50],[0.45, 0.55]])
# The biases for the first hidden layer
biases_out = np.array([0.6, 0.6]).reshape(2,1)
# Compute the intermediate value "z" for the output layer
z_out = np.dot(wts_2, actvn_hid) + biases_out
# Compute the activations for the output layer.
actvn_out = sigmoid(z_out)

# Define the desired outputs (the "ground truth")
y_vals = np.array([0.01, 0.99])
# Calculate the gradient of C wrt actvn_out.
# The gradient function was obtained by taking the first derivative of the 
# cross-entropy cost wrt a (see above in problem 1.2 for this calculation).
gradient_C = np.array([-y_vals[0]/actvn_out[0] + (1-y_vals[0])/(1-actvn_out[0]),
                        -y_vals[1]/actvn_out[1] + (1-y_vals[1])/(1-actvn_out[1])]).reshape(2,1)
# Calculate the value of sigmoid' function (first derivative of sigmoid wrt z)
# evaluated with the z_out vector.
sig_prime_z_out = np.apply_along_axis(sig_prime, 1, z_out).reshape(2,1)


#==============================================================================
# Compute the errors of the output layer ==== THIS IS ONE OF THE ANSWERS
#==============================================================================
delta_out = np.multiply(gradient_C, sig_prime_z_out)
delta_out

# By defn, delta_hid = [t(w^out)][delta_out] hadamard [sig_prime_z_hid]
# where t(w^out) is the transpose of wts_2
sig_prime_z_hid = np.apply_along_axis(sig_prime, 0, z_hid).reshape(2,1)


#==============================================================================
# Compute the errors of the hidden layer ===== THIS IS ONE OF THE ANSWERS
#==============================================================================
delta_hid = np.multiply(np.matmul(np.transpose(wts_2), delta_out), sig_prime_z_hid)
delta_hid


# By the 4th Fundamental Equation of Backpropogation, we know that:
# (partial C/partial w_{j,k}^l) = a_{k}^{l-1} * delta_j^l
# In other words, the partial wrt to some weight is equal to the activation
# in the previous layer * the error in the next layer (where the weight is
# connecting the activation neuron in {l-1} and the delta neuron in {l}).
#==== THESE ARE ALL ANSWERS ===================================================
partial_w1 = inputs[0] * delta_hid[0]
partial_w2 = inputs[0] * delta_hid[1]
partial_w3 = inputs[1] * delta_hid[0]
partial_w4 = inputs[1] * delta_hid[1]
partial_w5 =actvn_hid[0] * delta_out[0]
partial_w6 =actvn_hid[0] * delta_out[1]
partial_w7 =actvn_hid[1] * delta_out[0]
partial_w8 =actvn_hid[1] * delta_out[1]

partials_wrt_weights = np.array([partial_w1,
                                 partial_w2,
                                 partial_w3,
                                 partial_w4,
                                 partial_w5,
                                 partial_w6,
                                 partial_w7,
                                 partial_w8])
partials_wrt_weights


# By the 3rd Fundamental Equation of Backpropogation, we know that:
# (partial C/partial b_j^l) = delta_j^l
# Per our notation, this means that 
#==== THESE ARE ALL ANSWERS ===================================================
partial_b_hid_1 = delta_hid[0]
partial_b_hid_2 = delta_hid[1]
partial_b_out_1 = delta_out[0]
partial_b_out_2 = delta_out[1]

partials_wrt_biases = np.array([partial_b_hid_1,
                                partial_b_hid_2,
                                partial_b_out_1,
                                partial_b_out_2])
partials_wrt_biases





