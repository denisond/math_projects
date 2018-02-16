import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def error_function(line_var, data):
    """This function takes in a line (array with a slope and y-int), \
    and a data set (2d array with coordinates for all of the data points in our scatter plot above). And it returns single value\
    as an error term."""
    sum_of_squares_diffs = np.sum((data[:, 1] - (line_var[0] * data[:, 0] + line_var[1])) ** 2)
    return sum_of_squares_diffs


def fit_line(data, error_fcn):
    "Takes in a data set(2d array with coordinates) and an error function, and returns the line(2d-array with slope and y-int)\
    that minimizes the error term."
    guess_line = np.array([0, np.mean(data[:, 1])])

    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, guess_line[0] * x_ends + guess_line[1], 'm--', linewidth=2.0, label='Initial guess')

    result = spo.minimize(error_function, guess_line, args=(data,), method='SLSQP', options={'disp': True})
    return result.x


def run_experiment():
    # create and plot initial line.
    original_line = np.array(
        [4, 2])  # We will fix our slope as m = 4, and our y-intercept as b = 2, accomplished here with this array.
    original_Xvalues = np.array([x * 0.5 for x in range(0, 20)])
    original_Yvalues = original_line[0] * original_Xvalues + original_line[1]
    fig = plt.figure(figsize=(8, 8))
    plt.plot(original_Xvalues, original_Yvalues, label='Original Line')

    # add data points around original line.
    std_devs = 10
    noise = np.random.normal(0, std_devs, len(original_Yvalues))
    data = np.array([original_Xvalues, original_Yvalues + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    # call function to fit data.
    l_fit = fit_line(data, error_function)
    print('Fitted line: m = {}, b = {}'.format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label='fitted line')
    plt.legend(['Original line', 'Data points', 'Initial Guess', 'Fitted Line'])
    plt.show()

run_experiment()

"""NOW SWITCH TO FITTING A POLYNOMIAL TO DATA"""


def error_function_poly(coefs, data):  # error function
    """Parameters:
    coefs: coefficients for our polynomial guess (numpy.poly1d object).
    data: 2d-array with coordinates for all of the data points in our scatter plot above

    Returns single value as an error term."""
    sum_of_squared_Y_diffs = np.sum((data[:, 1] - np.polyval(coefs, data[:, 0])) ** 2)
    return sum_of_squared_Y_diffs

def fit_poly(data, error_func, degree=3):
    """Fit a line to given data, using a supplied error fcn.
    data: 2d array where each row is a point(X, Y)
    error_func: function that computes the error between a polynomial and observed data

    Returns polynomial that minimizes the error fcn.
    """

    # Generate initial guess for polynomoial
    C_guess = np.poly1d(np.ones(degree+1,dtype=np.float32)) # m=0,b=mean(y_values)
    print('\n',"C_guess:",C_guess)

    # plot initial guess (optional)
    x = np.linspace(-2, 5, 21)
    plt.plot(x, np.polyval(C_guess, x), 'm--', linewidth=2.0, label="Initial guess")

    # Call optimizer to minimize error fcn
    result=spo.minimize(error_func, C_guess, args=(data,), method='SLSQP', options={'disp': True})
    return np.poly1d(result.x)

def run_poly_experiment():
    # Define original polynomial with array of coefficients
    p_orig = np.float32([2,3,-2,4])
    Xorig = np.linspace(-5, 5, 21)
    Yorig = np.poly1d(p_orig)(Xorig)
    print('Original Line:', np.poly1d(p_orig))
    fig = plt.figure(figsize=(10,10))
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    # Generate noisy data points
    noise_sigma =40.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:, 1], 'go', label="Data points")

    # Try to fit a line to this data
    p_fit = fit_poly(data, error_function_poly)
    print('pfit:{}'.format(p_fit))
    print('Fitted line: C0 = {}, C1 = {}'.format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0],p_fit(data[:,0]), 'r--', linewidth=2.0, label='fitted line')
    plt.legend(['Original line', 'Data points', 'Initial Guess', 'Fitted Line'])
    plt.show()

run_poly_experiment()