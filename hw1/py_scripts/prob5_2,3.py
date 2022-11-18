'''
    File name: prob5_2_3.py
    Author: Jared Hansen
    Date created: 01/15/2019
    Date last modified: 01/18/2019
    Python Version: 3.6
'''


'''
THIS IS THE CODE FOR PROBLEM 5.1 

PROMPT: Write code in Python with the file name prob2_2.py that randomly
        generates N points sampled uniformly in the interval x in [-1,3]. Then
        output the function y = x^2 -3x + 1 for each of the points generated.
        Then write code that adds zero-mean Gaussian noise with standard 
        deviation sigma to y. Make plots of x and y with N in {15,100}, and
        sigma in {0, 0.05, 0.2} (there should be six plots in total). Save the
        point sets for the following question.
        HINT: You may want to check the NumPy library for generating noise.
'''




#==============================================================================
#==== Import statements =======================================================
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import random

# Setting the seed for the program
random.seed(2345)





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


#=== I don't actually use this function because I couldn't figure out how to
#=== get it to play nicely with the plt.subplot() function. As such, I just
#=== ended up hard-coding things (can be seen at the bottom of the file).
def gen_plot(x_pts, y_pts, sigma):
    """
    This function creates a scatter plot of given arrays.
    
    Parameters:
        x_pts (numpy array): the x values being plotted, used to generate y's
        y_pts (numpy array): the y values being plotted (noised-up)
        sigma (float): the standard deviation used to add Gaussian noise to the
                       y values.
    
    Returns:
        Nothing.
    """
    N = len(x_pts)
    plt.scatter(x_pts, y_pts)
    title = "Scatterplot for N=" + str(N) + " and sigma=" + str(sigma) 
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("noised-up y")
   
    
    
 
    
#==============================================================================
#==== Procedural programming: generating and plotting data ====================
#==============================================================================
    
# We only need one set of the true x values and y values of size 15.
x_n15 = gen_x_vals(15)
y_n15 = gen_y_vals(x_n15)
# Noised-up y points for N = 15, sigma = 0
# By necessity, this will be identical to the y_n15 array.
y_n15_sig0 = add_noise(0, y_n15)
# Noised-up y points for N = 15, sigma = 0.05
y_n15_sig005 = add_noise(0.05, y_n15)
# Noised-up y points for N = 15, sigma = 0.2
y_n15_sig02 = add_noise(0.2, y_n15)

# We only need one set of the true x values and y values of size 100.
x_n100 = gen_x_vals(100)
y_n100 = gen_y_vals(x_n100)
# Noised-up y points for N = 100, sigma = 0
# By necessity, this will be identical to the y_n100 array.
y_n100_sig0 = add_noise(0, y_n100)
# Noised-up y points for N = 100, sigma = 0.05
y_n100_sig005 = add_noise(0.05, y_n100)
# Noised-up y points for N = 100, sigma = 0.2
y_n100_sig02 = add_noise(0.2, y_n100)


#==== Plotting all 6 combinations of N and sigma in a single plot =============
plt.figure(1)

plt.subplot(2, 3, 1)
plt.scatter(x_n15, y_n15_sig0, s = 3)
plt.title("N=15, sigma=0")
plt.xlabel("x values")
plt.ylabel("noised-up y")

plt.subplot(2, 3, 2)
plt.scatter(x_n15, y_n15_sig005, s = 3)
plt.title("N=15, sigma=0.05")
plt.xlabel("x values")
plt.ylabel("noised-up y")

plt.subplot(2, 3, 3)
plt.scatter(x_n15, y_n15_sig02, s = 3)
plt.title("N=15, sigma=0.2")
plt.xlabel("x values")
plt.ylabel("noised-up y")

plt.subplot(2, 3, 4)
plt.scatter(x_n100, y_n100_sig0, s = 3)
plt.title("N=100, sigma=0")
plt.xlabel("x values")
plt.ylabel("noised-up y")

plt.subplot(2, 3, 5)
plt.scatter(x_n100, y_n100_sig005, s = 3)
plt.title("N=100, sigma=0.05")
plt.xlabel("x values")
plt.ylabel("noised-up y")

plt.subplot(2, 3, 6)
plt.scatter(x_n100, y_n100_sig02, s = 3)
plt.title("N=100, sigma=0.2")
plt.xlabel("x values")
plt.ylabel("noised-up y")

plt.subplots_adjust(top=0.85, bottom=0.12, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)

plt.suptitle("Plots of X values versus Y values with noise added")
plt.show()








































'''
THIS IS MY CODE FOR PROBLEM 5.3

    File name: prob5_2_3.py
    Author: Jared Hansen
    Date created: 01/16/2019
    Date last modified: 01/18/2019
    Python Version: 3.6
'''


'''
PROMPT: Find the optimal weights (in terms of MSE) for fitting a polynomial
        function to the data in all 6 cases generated above using a polynomial
        of degree 1, 2, and 9. Use the equation given above. Include your code
        in prob2_3.py. Do not use built-in methods for regression. Plot the
        fitted curves on the same plot as the data points (you can plot  all 3
        polynomial curves on the same plot.) Report the fitted weights and the
        MSE in tables. Do any of the models overfit or underfit the data?
'''

#==============================================================================
#==== Import statements =======================================================
#==============================================================================

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt





#==============================================================================
#==== Function definitions ====================================================
#==============================================================================

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


def calc_wts(x_mat, y_arr, deg_poly):
    """
    This function calculates the weights to minimize MSE for polynomial 
    regression.
    
    Parameters:
        x_mat (numpy matrix): needed matrix of values for regression. This 
        matrix CANNOT have the column of 1's appended to the right side yet.
        y_arr (numpy array): nx1 array of y values that have noise added.
        deg_poly (int): the degree of the polynomial to be used for regression.
        
    Returns:
        wts (numpy array): the array of optimal weights for the regression.
    """
    x_mat = new_x_matrix(x_mat, deg_poly)
    wts = np.dot(LA.inv(np.dot(x_mat.T, x_mat)),
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


def calc_mse(orig_x_arr, noised_y, deg_poly):
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
    opt_wts = calc_wts(orig_x_arr, noised_y, deg_poly)
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
#==== Procedural Programming ==================================================
#==============================================================================

# Python has a hard time performing some of the calculations with the 1D
# arrays left as they are. I'm going to manually convert them to a format that
# will work with numpy matrix algebra operations.
x_n15         = x_n15.reshape(len(x_n15), 1)
x_n100        = x_n100.reshape(len(x_n100), 1)
y_n15         = y_n15.reshape(len(y_n15), 1)
y_n15_sig0    = y_n15_sig0.reshape(len(y_n15_sig0), 1)
y_n15_sig005  = y_n15_sig005.reshape(len(y_n15_sig005), 1)
y_n15_sig02   = y_n15_sig02.reshape(len(y_n15_sig02), 1)
y_n100        = y_n100.reshape(len(y_n100), 1)
y_n100_sig0   = y_n100_sig0.reshape(len(y_n100_sig0), 1)
y_n100_sig005 = y_n100_sig005.reshape(len(y_n100_sig005), 1)
y_n100_sig02  = y_n100_sig02.reshape(len(y_n100_sig02), 1)

# We'll use a vector of 1000 points for plotting the fitted polynomial curves.
# This vector will be used for those curves fitted on the N=15 data.
x_n15_1000 = np.linspace(min(x_n15), max(x_n15), 1000)
x_n15_1000 = x_n15_1000.reshape(len(x_n15_1000), 1)


#=== Weights, MSE, fctn for: N = 15, sigma = 0, poly = 1
wts_n15_sig0_p1 = calc_wts(x_n15, y_n15_sig0, 1)
mse_n15_sig0_p1 = calc_mse(x_n15, y_n15_sig0, 1)
y_n15_s0_1000_p1 = gen_y_hats(x_n15_1000, wts_n15_sig0_p1, 1)
#=== Weights, MSE, fctn for: N = 15, sigma = 0, poly = 2
wts_n15_sig0_p2 = calc_wts(x_n15, y_n15_sig0, 2)
mse_n15_sig0_p2 = calc_mse(x_n15, y_n15_sig0, 2)
y_n15_s0_1000_p2 = gen_y_hats(x_n15_1000, wts_n15_sig0_p2, 2)
#=== Weights, MSE, fctn for: N = 15, sigma = 0, poly = 9
wts_n15_sig0_p9 = calc_wts(x_n15, y_n15_sig0, 9)
mse_n15_sig0_p9 = calc_mse(x_n15, y_n15_sig0, 9)
y_n15_s0_1000_p9 = gen_y_hats(x_n15_1000, wts_n15_sig0_p9, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n15, y_n15_sig0, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s0_1000_p1, c = "r",
         label = "$f_1(x) = -0.73x + 1.00$")
plt.plot(x_n15_1000, y_n15_s0_1000_p2, c = "g",
         label = "$f_2(x) = f_9(x) = -3x + x^2 +1$")
#plt.plot(x_n15_1000, y_n15_s0_1000_p9, c = "k", label = "$f_9(x) = -3x + x^2 + 1$")
plt.title("N=15, sigma=0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 1
wts_n15_sig005_p1 = calc_wts(x_n15, y_n15_sig005, 1)
mse_n15_sig005_p1 = calc_mse(x_n15, y_n15_sig005, 1)
y_n15_s005_1000_p1 = gen_y_hats(x_n15_1000, wts_n15_sig005_p1, 1)
#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 2
wts_n15_sig005_p2 = calc_wts(x_n15, y_n15_sig005, 2)
mse_n15_sig005_p2 = calc_mse(x_n15, y_n15_sig005, 2)
y_n15_s005_1000_p2 = gen_y_hats(x_n15_1000, wts_n15_sig005_p2, 2)
#=== Weights, MSE, fctn for: N = 15, sigma = 0.05, poly = 9
wts_n15_sig005_p9 = calc_wts(x_n15, y_n15_sig005, 9)
mse_n15_sig005_p9 = calc_mse(x_n15, y_n15_sig005, 9)
y_n15_s005_1000_p9 = gen_y_hats(x_n15_1000, wts_n15_sig005_p9, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n15, y_n15_sig005, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s005_1000_p1, c = "r",
         label = "$f_1(x) = -0.73x + 1.01$")
plt.plot(x_n15_1000, y_n15_s005_1000_p2, c = "g",
         label = "$f_2(x) = -3.01x + 1.01x^2 +1.01$")
plt.plot(x_n15_1000, y_n15_s005_1000_p9, c = "k",
         label = "$f_9(x) = -3.04x + 0.71x^2 + 0.45x^3 + 0.18x^4 - 1.01x^5 + 1.06x^6 - 0.52x^7 + 0.13x^8 - 0.01x^9 + 1.05 $")
plt.title("N=15, sigma=0.05")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


#=== Weights, MSE, fctn for: N = 15, sigma = 0.2, poly = 1
wts_n15_sig02_p1 = calc_wts(x_n15, y_n15_sig02, 1)
mse_n15_sig02_p1 = calc_mse(x_n15, y_n15_sig02, 1)
y_n15_s02_1000_p1 = gen_y_hats(x_n15_1000, wts_n15_sig02_p1, 1)
#=== Weights, MSE, fctn for: N = 15, sigma = 0.2, poly = 2
wts_n15_sig02_p2 = calc_wts(x_n15, y_n15_sig02, 2)
mse_n15_sig02_p2 = calc_mse(x_n15, y_n15_sig02, 2)
y_n15_s02_1000_p2 = gen_y_hats(x_n15_1000, wts_n15_sig02_p2, 2)
#=== Weights, MSE, fctn for: N = 15, sigma = 0.2, poly = 9
wts_n15_sig02_p9 = calc_wts(x_n15, y_n15_sig02, 9)
mse_n15_sig02_p9 = calc_mse(x_n15, y_n15_sig02, 9)
y_n15_s02_1000_p9 = gen_y_hats(x_n15_1000, wts_n15_sig02_p9, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n15, y_n15_sig02, s = 20, c = "b", label = "data")
plt.plot(x_n15_1000, y_n15_s02_1000_p1, c = "r",
         label = "$f_1(x) = -0.74x + 0.93$")
plt.plot(x_n15_1000, y_n15_s02_1000_p2, c = "g",
         label = "$f_2(x) = -3.24x + 1.10x^2 + 0.93 $")
plt.plot(x_n15_1000, y_n15_s02_1000_p9, c = "k",
         label = "$f_9(x) = -4.03x + 0.42x^2 + 5.17x^3 + -2.63x^4 -6.87x^5 + 9.82x^6 - 5.24x^7 + 1.28x^8 - 0.12x^9 + 1.03 $")
plt.title("N=15, sigma=0.2")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show



# We'll use a vector of 1000 points for plotting the fitted polynomial curves.
# This vector will be used for those curves fitted on the N=15 data.
x_n100_1000 = np.linspace(min(x_n100), max(x_n100), 1000)
x_n100_1000 = x_n100_1000.reshape(len(x_n100_1000), 1)

#=== Weights, MSE, fctn for: N = 100, sigma = 0, poly = 1
wts_n100_sig0_p1 = calc_wts(x_n100, y_n100_sig0, 1)
mse_n100_sig0_p1 = calc_mse(x_n100, y_n100_sig0, 1)
y_n100_s0_1000_p1 = gen_y_hats(x_n100_1000, wts_n100_sig0_p1, 1)
#=== Weights, MSE, fctn for: N = 100, sigma = 0, poly = 2
wts_n100_sig0_p2 = calc_wts(x_n100, y_n100_sig0, 2)
mse_n100_sig0_p2 = calc_mse(x_n100, y_n100_sig0, 2)
y_n100_s0_1000_p2 = gen_y_hats(x_n100_1000, wts_n100_sig0_p2, 2)
#=== Weights, MSE, fctn for: N = 100, sigma = 0, poly = 9
wts_n100_sig0_p9 = calc_wts(x_n100, y_n100_sig0, 9)
mse_n100_sig0_p9 = calc_mse(x_n100, y_n100_sig0, 9)
y_n100_s0_1000_p9 = gen_y_hats(x_n100_1000, wts_n100_sig0_p9, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n100, y_n100_sig0, s = 10, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s0_1000_p1, c = "r",
         label = "$f_1(x) = -0.93x + 1.31$")
plt.plot(x_n100_1000, y_n100_s0_1000_p2, c = "g",
         label = "$f_2(x) = f_9(x) = -3x + x^2 +1$")
#plt.plot(x_n15_1000, y_n15_s0_1000_p9, c = "k", label = "$f_9(x) = -3x + x^2 + 1$")
plt.title("N=100, sigma=0")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


#=== Weights, MSE, fctn for: N = 100, sigma = 0.05, poly = 1
wts_n100_sig005_p1 = calc_wts(x_n100, y_n100_sig005, 1)
mse_n100_sig005_p1 = calc_mse(x_n100, y_n100_sig005, 1)
y_n100_s005_1000_p1 = gen_y_hats(x_n100_1000, wts_n100_sig005_p1, 1)
#=== Weights, MSE, fctn for: N = 100, sigma = 0.05, poly = 2
wts_n100_sig005_p2 = calc_wts(x_n100, y_n100_sig005, 2)
mse_n100_sig005_p2 = calc_mse(x_n100, y_n100_sig005, 2)
y_n100_s005_1000_p2 = gen_y_hats(x_n100_1000, wts_n100_sig005_p2, 2)
#=== Weights, MSE, fctn for: N = 100, sigma = 0.05, poly = 9
wts_n100_sig005_p9 = calc_wts(x_n100, y_n100_sig005, 9)
mse_n100_sig005_p9 = calc_mse(x_n100, y_n100_sig005, 9)
y_n100_s005_1000_p9 = gen_y_hats(x_n100_1000, wts_n100_sig005_p9, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n100, y_n100_sig005, s = 10, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s005_1000_p1, c = "r",
         label = "$f_1(x) = -0.93x + 1.31$")
plt.plot(x_n100_1000, y_n100_s005_1000_p2, c = "g",
         label = "$f_2(x) = -3.005x + 1.000x^2 +1.002$")
plt.plot(x_n100_1000, y_n100_s005_1000_p9, c = "k",
         label = "$f_9(x) = -2.98x + 8.93x^2 -5.98x^3 + 2.13x^4 -3.13x^5 -1.15x^6 + 7.48x^7 -1.73x^8 + 1.38x^9 + 1.01 $")
plt.title("N=100, sigma=0.05")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show


#=== Weights, MSE, fctn for: N = 100, sigma = 0.2, poly = 1
wts_n100_sig02_p1 = calc_wts(x_n100, y_n100_sig02, 1)
mse_n100_sig02_p1 = calc_mse(x_n100, y_n100_sig02, 1)
y_n100_s02_1000_p1 = gen_y_hats(x_n100_1000, wts_n100_sig02_p1, 1)
#=== Weights, MSE, fctn for: N = 100, sigma = 0.2, poly = 2
wts_n100_sig02_p2 = calc_wts(x_n100, y_n100_sig02, 2)
mse_n100_sig02_p2 = calc_mse(x_n100, y_n100_sig02, 2)
y_n100_s02_1000_p2 = gen_y_hats(x_n100_1000, wts_n100_sig02_p2, 2)
#=== Weights, MSE, fctn for: N = 100, sigma = 0.2, poly = 9
wts_n100_sig02_p9 = calc_wts(x_n100, y_n100_sig02, 9)
mse_n100_sig02_p9 = calc_mse(x_n100, y_n100_sig02, 9)
y_n100_s02_1000_p9 = gen_y_hats(x_n100_1000, wts_n100_sig02_p9, 9)
#=== Plot all three of these polynomial approximations on the same plot.
plt.scatter(x_n100, y_n100_sig02, s = 10, c = "b", label = "data")
plt.plot(x_n100_1000, y_n100_s02_1000_p1, c = "r",
         label = "$f_1(x) = -0.92x + 1.30$")
plt.plot(x_n100_1000, y_n100_s02_1000_p2, c = "g",
         label = "$f_2(x) = -2.98x + 0.99x^2 + 0.99$")
plt.plot(x_n100_1000, y_n100_s02_1000_p9, c = "k",
         label = "$f_9(x) = -2.82x + 0.92x^2 -0.65x^3 + 0.58x^4 + 0.38x^5 -0.63x^6 + 2.60x^7 -0.04x^8 + 0.0008x^9 + 1.00 $")
plt.title("N=100, sigma=0.2")
plt.ylabel("noised-up y")
plt.xlabel("x values")
plt.grid()
plt.legend()
plt.show