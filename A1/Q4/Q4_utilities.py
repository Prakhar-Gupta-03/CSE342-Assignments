import random
import math
import numpy as np
import matplotlib.pyplot as plt

def combine_vectors(vector1, vector2):
    combined_matrix = np.vstack((vector1, vector2))
    combined_matrix = combined_matrix.T
    return combined_matrix

def sinosuidal_wave_generation(amplitude, frequency, phase, time):
    wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return wave

def ramp_wave(t,period):
    t = t % period
    return (t/period) - 1

def ramp_wave_generation(time, period):
    generated_ramp_wave = ramp_wave(time, period)
    return generated_ramp_wave

#centering a vector
def center_vector(x):
    mean = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        # mean[i] = np.mean(x[i,:])
        for j in range(x.shape[1]):
            mean[i] += x[i,j]
        mean[i] /= x.shape[1]
    centered_x = x - mean
    return centered_x, mean  

def covariance(x):
    # centering the matrix x
    centered_matrix, mean = center_vector(x)
    # number of columns in the matrix
    no_of_columns = np.shape(x)[1] - 1
    # calculating the covariance matrix
    # formula for covariance matrix is: 1/n * (centered matrix) * (centered matrix)^T
    covariance_matrix = np.matmul(centered_matrix, centered_matrix.T) / no_of_columns
    return covariance_matrix


def whitening(X):
    # Calculate the covariance matrix of the given matrix 

    covariance_matrix = covariance(X)
    print(covariance_matrix)
    # Decomposing the covariance matrix into eigenvalues and eigenvectors using single value decomposition
    # matrix U contains the eigenvectors
    # matrix S contains the eigenvalues

    U, S, V = np.linalg.svd(covariance_matrix)

    # now that we have the eigenvalues, we need to calculate the diagonal matrix to the power of -1/2
    # first we need to take the square root of the eigenvalues

    square_root_of_eigenvalues = np.sqrt(S)

    # then we need to take the inverse of the square root of the eigenvalues

    inverse_of_square_root_of_eigenvalues = 1/square_root_of_eigenvalues

    # then we need to create a diagonal matrix with the inverse of the square root of the eigenvalues
    
    diagonal_matrix = np.diag(inverse_of_square_root_of_eigenvalues)
    diagonal_matrix = np.diag(1.0 / np.sqrt(S))
    #now we need to calculate the whitening matrix of the given matrix 
    #formula for whitening matrix is: 
    #whitening_matrix = U * (eigenvalues)^-1/2 * U^T
    #whitening_matrix = U * (diagonal_matrix) * U^T

    whitening_matrix = np.matmul(U, np.matmul(diagonal_matrix, U.T))

    #now we need to project the whitened matrix onto the whitening matrix
    #formula for whitened matrix is:
    #whitened_matrix = whitening_matrix * (original matrix)
    whitened_matrix = np.matmul(whitening_matrix, X)
    #return the whitened matrix and the whitening matrix
    return whitened_matrix, whitening_matrix

def fastIca(signals,  alpha = 1, threshold=1e-8, max_iterations=10000):
    no_of_rows = signals.shape[0]
    no_of_columns = signals.shape[1]
    # Initialize random weights
    weights = np.random.rand(no_of_rows, no_of_rows)

    for c in range(no_of_rows):
            weight = weights[c, :].copy().reshape(no_of_rows, 1)        # Get the weights for the current component
            weight = weight / np.sqrt((weight ** 2).sum())              # Normalize the weights

            i = 0
            max = 100
            while ((max > threshold) & (i < max_iterations)):           # Iterate until maximum is less than threshold or max iterations is reached

                # Dot product of weight and signal
                weight_signals = np.dot(weight.T, signals)              

                # Pass w*s into contrast function g
                tanh_weight_signals = np.tanh(weight_signals * alpha)   # tanh is the contrast function
                weight_contrast_function = tanh_weight_signals.T        # Transpose the contrast function

                # Pass w*s into g prime
                tan_weight = np.tanh(weight_signals)                    # tanh is the contrast function
                square_tan_weight = np.square(tan_weight)               
                difference = 1 - square_tan_weight                
                difference = difference * alpha                   
                weight_contrast_function_derivative = difference        # Derivative of contrast function

                # Update weights
                weight_contrast_function_signal = signals * weight_contrast_function.T                                 
                weight_contrast_function_signal_mean = weight_contrast_function_signal.mean(axis=1)
                weight_contrast_function_derivative_mean = weight_contrast_function_derivative.mean()
                weight_contrast_function_derivative_mean = weight_contrast_function_derivative_mean * weight.squeeze()
                weights_updated = weight_contrast_function_signal_mean - weight_contrast_function_derivative_mean          # Update weights to be used in next iteration

                # Decorrelate weights              
                weights_segment = weights[:c]
                weights_segment = weights_segment.T
                weights_multiply = np.dot(weights_updated, weights_segment)
                weights_multiply = np.dot(weights_multiply, weights_segment.T)
                weights_updated = weights_updated - weights_multiply                                                    # Decorrelate weights
                weights_updated = weights_updated / np.sqrt((weights_updated ** 2).sum())                               # Normalize weights

                # Calculate maximum difference between weights
                weights_max = weights_updated * weight                              # Dot product of weights
                weights_max = weights_max.sum()                                     # Dot product of weights
                weights_max = np.abs(weights_max)                                   # Absolute value of weights
                weights_max = np.abs(weights_max - 1)                               # Absolute value of difference between weights
                max = weights_max                                                   

                # Update weights
                weight = weights_updated

                # Update counter
                i += 1

            weights[c, :] = weight.T
    return weights