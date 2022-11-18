'''
    File name: prob1.py
    Author: Jared Hansen
    Date created: 01/22/2019
    Date last modified: 01/22/2019
    Python Version: 3.6
'''


'''
CODE FOR HW2, PROLEM 1 IN [DEEP LEARNING: THEORY AND APPLICATIONS]

PROMPT: Consider the setting we had in problem 5 in Homework 1. In this
        problem, you will reuse the code you wrote in problem 5.2 to generate
        data from the same model. Fit a 9-degree polynomial model with L2 norm
        regularization to the cases with sigma=0.05 and N in {15,100}, and
        include your code in prob1.py. Vary the parameter lambda, and choose
        three values of lambda that result in the following scenarios: 
        underfitting, overfitting, and an appropriate fit. Report the fitted
        weights and the MSE in each of these scenarios.
HINT: linear regression with L2 norm regularization is also referred to as
      ridge regression. You may find the equation for this in the machine
      learning slides.
'''




#==============================================================================
#==== Import statements =======================================================
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA

# Setting the seed for the program
np.random.seed(2345)





#==============================================================================
#==== Function definitions ====================================================
#==============================================================================
def gen_x_vals(n_points):
    """
    Generates x values by sampling uniformly on the interval [-1,3]. Since the
    np.random.uniform(a,b) function only samples from [a,b) I got around this
    by rounding each of the numbers sampled points to 10 decimal places. That
    way there is the possibility of getting a number at the upper bound 3.
    
    Parameters:
        n_points (int): the number of points desired in the sample.
        
    Returns:
        x_vals (numpy array): the NumPy array of the sampled points.
    """
    x_vals = np.round_(np.random.uniform(low = -1.0,
                                        high = 3.0,
                                        size = n_points), 10)
    return x_vals


def gen_y_vals(x_array):
    """
    Generates and returns a NumPy array of y values by taking an array of x
    values as input. Uses the function y = x**2 - 3*x + 1 to generate the y's.
    
    Parameters:
        x_array (numpy array): NumPy array of x values.
        
    Returns:
        y_array (numpy array): NumPy array of y values. 
    """
    y_array = np.round(x_array**2 - 3*x_array + 1, 10)
    return y_array
    

def add_noise(sigma, y_array):
    """
    This function adds Gaussian noise of mean 0 and standard deviation sigma to
    each of the values in the y_array.
    
    Parameters:
        sigma (float): the standard deviation of the Gaussian noise.
        y_array (numpy array): NumPy array of y values.
        
    Returns:
        noised_y (numpy array): NumPy array of the now-noised-up y values.
    """
    noised_y = y_array + np.random.normal(loc = 0,
                                          scale = sigma,
                                          size = len(y_array))
    return noised_y


def new_x_matrix(x_arr, deg_poly):
    """
    This function takes a 1D array and degree of polynomial as inputs and
    returns the correct X matrix for performing polynomial regression.
    
    Parameters:
        x_arr (numpy array): randomly generated x values.
        deg_poly (int): the degree of the polynomial to be used for regression.
        
    Returns:
        new_mat (numpy matrix): the needed matrix to perform polynomial
        regression for the degree of polynomial specified.
    """
    new_mat = x_arr
    # Append columns to create new X matrix
    if(deg_poly > 1):
        for i in range(2, deg_poly+1, 1):
            new_mat = np.asmatrix(np.column_stack((new_mat, x_arr**i)))
    # Append a column of 1's to the end of the matrix
    new_mat = np.asmatrix(np.column_stack((new_mat, np.ones(len(x_arr)))))
    return new_mat


def create_id_mat(deg_poly, lamb_val):
    """
    This function creates an identity matrix that is multiplied by the scalar
    lambda (lamb_val). This matrix is used in calculation of the optimal
    weights for L2-norm regularized regression, and has dimension
    [deg_poly + 1]x[deg_poly +1].

    Parameters
    ----------
    deg_poly : int
        This is the degree of the polynomial approximation being used for 
        regression. Will only be 9 for this problem, could be changed though.
    lamb_val : int
        The desired value of lambda for the regularization term. This will be
        varied to induce overfitting, underfitting, and an appropriate fit.
    
    Returns
    -------
    id_mat : numpy matrix
        The identity matrix of dim [deg_poly + 1]x[deg_poly +1] that is
        multiplied by the scalar lamb_val.
    
    """
    lamb_vec = np.full(deg_poly+1, lamb_val)
    id_mat = np.diag(lamb_vec)
    return id_mat


def calc_wts(x_mat, y_arr, deg_poly, lamb_val):
    """
    This function calculates the weights to minimize MSE for polynomial 
    regression with added L2 norm regularization term
    
    Parameters
    ----------
        x_mat : numpy matrix
            needed matrix of values for regression. This matrix CANNOT have the
            column of 1's appended to the right side yet.
        y_arr : numpy array
            nx1 array of y values that have noise added.
        deg_poly : int
            the degree of the polynomial to be used for regression.
        lamb_val : int
            the value of lambda used for regularization.
        
        
    Returns
    -------
        wts (numpy array): the array of optimal weights for the regression.
    """
    x_mat = new_x_matrix(x_mat, deg_poly)
    id_mat = create_id_mat(deg_poly, lamb_val)
    wts = np.dot((LA.inv(id_mat + np.dot(x_mat.T, x_mat))),
                 np.dot(x_mat.T, y_arr.reshape(len(y_arr), 1)))
    return wts


def gen_y_hats(x_arr, wts, deg_poly):
    """
    This function generates the predicted response values for a given vector of
    x value inputs, optimal weights, and degree of polynomial function.
    
    Parameters:
        x_arr (numpy array): the original 1D array of x values.
        wts (numpy array): optimal weights for the regression.
        deg_poly (int): the degree of the polynomial to be used for regression.
        
    Returns:
        y_hats (numpy array): array of predicted y values (response values).
    """
    x_mat = new_x_matrix(x_arr, deg_poly)
    y_hats = np.dot(x_mat, wts)
    return y_hats


def calc_mse(orig_x_arr, noised_y, deg_poly, lamb_val):
    """
    This function calculates the MSE for a given array of input (x) values, 
    noised-up y values, and specified degree polynomial.
    
    Parameters:
        orig_x_arr (numpy array): original array of x values (1-dimensional).
        noised_y (numpy array): array of noised-up y values.
        deg_poly (int): the degree of the polynomial to be use for regression.
        
    Returns:
        mse (float): value of the MSE for a given regression. This is found by
        multiplying (X^T X)^-1 (X^T y) and scaling by 1/n.
    """
    
    # true_ys: 1d array
    # pred_ys: 1d array (same length)
    opt_wts = calc_wts(orig_x_arr, noised_y, deg_poly, lamb_val)
    pred_ys = gen_y_hats(orig_x_arr, opt_wts, deg_poly)
    if(len(pred_ys) == 15):
        true_ys = y_n15
    else:
        true_ys = y_n100
    diffs = true_ys - pred_ys
    diffs_sqrd = np.square(diffs)
    mse = np.sum(diffs_sqrd) / len(pred_ys)
    return mse



   
    
    
 
    
#==============================================================================
#==== Procedural programming: generating data, fitting models =================
#==============================================================================


#==== GENERATING THE DESIRED DATA
    
# True x values and y values of size N=15.
x_n15 = gen_x_vals(15)
y_n15 = gen_y_vals(x_n15)

# True x values and y values of size N=100.
x_n100 = gen_x_vals(100)
y_n100 = gen_y_vals(x_n100)

# Noised-up y points for N = 15, sigma = 0.05
y_n15_sig005 = add_noise(0.05, y_n15)
# Noised-up y points for N = 100, sigma = 0.05
y_n100_sig005 = add_noise(0.05, y_n100)


# =============================================================================
# #==== Plotting the generated data (making sure things worked correctly ======
# plt.figure(1)
# 
# plt.subplot(1, 2, 1)
# plt.scatter(x_n15, y_n15_sig005, s = 3)
# plt.title("N=15, sigma=0.05")
# plt.xlabel("x values")
# plt.ylabel("noised-up y")
# 
# plt.subplot(1, 2, 2)
# plt.scatter(x_n100, y_n100_sig005, s = 3)
# plt.title("N=100, sigma=0.05")
# plt.xlabel("x values")
# plt.ylabel("noised-up y")
# 
# plt.subplots_adjust(top=0.85, bottom=0.12, left=0.10, right=0.95, hspace=0.55,
#                     wspace=0.35)
# 
# plt.suptitle("Plots of X values versus Y values with noise added")
# plt.show()
# =============================================================================



#==== MODEL CREATION ==========================================================



# Python has a hard time performing some of the calculations with the 1D
# arrays left as they are. I'm going to manually convert them to a format that
# will work with numpy matrix algebra operations.
x_n15         = x_n15.reshape(len(x_n15), 1)
x_n100        = x_n100.reshape(len(x_n100), 1)
y_n15         = y_n15.reshape(len(y_n15), 1)
y_n15_sig005  = y_n15_sig005.reshape(len(y_n15_sig005), 1)
y_n100        = y_n100.reshape(len(y_n100), 1)
y_n100_sig005 = y_n100_sig005.reshape(len(y_n100_sig005), 1)

# We'll use a vector of 1000 points for plotting the fitted polynomial curves.
# This vector will be used for those curves fitted on the N=15 data.
x_n15_1000 = np.linspace(min(x_n15), max(x_n15), 1000)
x_n15_1000 = x_n15_1000.reshape(len(x_n15_1000), 1)

# We'll use a vector of 1000 points for plotting the fitted polynomial curves.
# This vector will be used for those curves fitted on the N=15 data.
x_n100_1000 = np.linspace(min(x_n100), max(x_n100), 1000)
x_n100_1000 = x_n100_1000.reshape(len(x_n100_1000), 1)


#=== N = 15 code ==============================================================
#==============================================================================

#=== lambda = 0 generates overfitting =========================================
#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 9, lambda = 0
wts_n15_sig005_p9_l0 = calc_wts(x_n15, y_n15_sig005, 9, lamb_val = 0)
mse_n15_sig005_p9_l0 = calc_mse(x_n15, y_n15_sig005, 9, lamb_val = 0)
y_n15_s005_1000_p9_l0 = gen_y_hats(x_n15_1000, wts_n15_sig005_p9_l0, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p9_l0, c = "r",
         label = "$f_9(x)$")
plt.title("OVERFITTING: N=15, sigma=0.05, lambda = 0.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

#=== lambda = 100 generates underfitting ======================================
#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 9, lambda = 100
wts_n15_sig005_p9_l100 = calc_wts(x_n15, y_n15_sig005, 9, lamb_val = 100)
mse_n15_sig005_p9_l100 = calc_mse(x_n15, y_n15_sig005, 9, lamb_val = 100)
y_n15_s005_1000_p9_l100 = gen_y_hats(x_n15_1000, wts_n15_sig005_p9_l100, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p9_l100, c = "r",
         label = "$f_9(x)$")
plt.title("UNDERFITTING: N=15, sigma=0.05, lambda=100.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

#=== lambda = 0.005 generates an appropriate fit ==============================
#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 9, lambda = 0.005
wts_n15_sig005_p9_l005 = calc_wts(x_n15, y_n15_sig005, 9, lamb_val = 0.005)
mse_n15_sig005_p9_l005 = calc_mse(x_n15, y_n15_sig005, 9, lamb_val = 0.005)
y_n15_s005_1000_p9_l005 = gen_y_hats(x_n15_1000, wts_n15_sig005_p9_l005, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p9_l005, c = "r",
         label = "$f_9(x)$")
plt.title("APPROPRIATE FIT: N=15, sigma=0.05, lambda = 0.005")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show



# This code plots all three polynomial models for N=15 in a single graphic.
plt.figure(1)

plt.subplot(1, 3, 1)
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p9_l0, c = "r",
         label = "$f_9(x)$")
plt.title("OVERFITTING: $\lambda$=0.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

plt.subplot(1, 3, 2)
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p9_l100, c = "r",
         label = "$f_9(x)$")
plt.title("UNDERFITTING: $\lambda$=100.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

plt.subplot(1, 3, 3)
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p9_l005, c = "r",
         label = "$f_9(x)$")
plt.title("APPROPRIATE FIT: $\lambda$=0.005")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

plt.subplots_adjust(top=0.85, bottom=0.12, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)

plt.suptitle("$9^{th}$-degree Polynomial Models: N=15, $\sigma$=0.05")
plt.show()














#=== N = 100 code =============================================================
#==============================================================================

#=== lambda = 0 generates overfitting =========================================
#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 9, lambda = 0
wts_n100_sig005_p9_l0 = calc_wts(x_n100, y_n100_sig005, 9, lamb_val = 0)
mse_n100_sig005_p9_l0 = calc_mse(x_n100, y_n100_sig005, 9, lamb_val = 0)
y_n100_s005_1000_p9_l0 = gen_y_hats(x_n100_1000, wts_n100_sig005_p9_l0, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n100, y_n100_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p9_l0, c = "r",
         label = "$f_9(x)$")
plt.title("OVERFITTING: N=100, sigma=0.05, lambda = 0.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


#=== lambda = 100 generates underfitting ======================================
#=== Weights, MSE, fctn for: N = 100, sigma = 0.05, poly = 9, lambda = 100
wts_n100_sig005_p9_l100 = calc_wts(x_n100, y_n100_sig005, 9, lamb_val = 100)
mse_n100_sig005_p9_l100 = calc_mse(x_n100, y_n100_sig005, 9, lamb_val = 100)
y_n100_s005_1000_p9_l100 = gen_y_hats(x_n100_1000, wts_n100_sig005_p9_l100, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n100, y_n100_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p9_l100, c = "r",
         label = "$f_9(x)$")
#plt.plot(x_n100_1000, y_n100_s0_1000_p9, c = "k", label = "$f_9(x) = -3x + x^2 + 1$")
plt.title("UNDERFITTING: N=100, sigma=0.05, lambda=100.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


#=== lambda = 0.005 generates an appropriate fit ==============================
#=== Weights, MSE, fctn for: N = 100, sigma = 0.05, poly = 9, lambda = 0.05
wts_n100_sig005_p9_l05 = calc_wts(x_n100, y_n100_sig005, 9, lamb_val = 0.05)
mse_n100_sig005_p9_l05 = calc_mse(x_n100, y_n100_sig005, 9, lamb_val = 0.05)
y_n100_s005_1000_p9_l05 = gen_y_hats(x_n100_1000, wts_n100_sig005_p9_l05, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n100, y_n100_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p9_l05, c = "r",
         label = "$f_9(x)$")
#plt.plot(x_n100_1000, y_n100_s0_1000_p9, c = "k", label = "$f_9(x) = -3x + x^2 + 1$")
plt.title("APPROPRIATE FIT: N=100, sigma=0.05, lambda = 0.05")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


# This code plots all three polynomial models for N=100 in a single graphic.
plt.figure(1)

plt.subplot(1, 3, 1)
plt.scatter(x_n100, y_n100_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p9_l0, c = "r",
         label = "$f_9(x)$")
plt.title("OVERFITTING: $\lambda$=0.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

plt.subplot(1, 3, 2)
plt.scatter(x_n100, y_n100_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p9_l100, c = "r",
         label = "$f_9(x)$")
plt.title("UNDERFITTING: $\lambda$=100.0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

plt.subplot(1, 3, 3)
plt.scatter(x_n100, y_n100_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p9_l05, c = "r",
         label = "$f_9(x)$")
plt.title("APPROPRIATE FIT: $\lambda$=0.05")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show

plt.subplots_adjust(top=0.85, bottom=0.12, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)

plt.suptitle("$9^{th}$-degree Polynomial Models: N=100, $\sigma$=0.05")
plt.show()
