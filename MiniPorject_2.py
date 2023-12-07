# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:08:49 2023

@author: jerem
"""

import numpy as np

import math
e = np.e

def gen_data(Gamma,n):
    inputs = np.random.uniform(-Gamma,Gamma,(n,3))
    y =  inputs[:, 0] * inputs[:, 1] + inputs[:,2]
    y = y.reshape(n,1)
    print(y)
    print(inputs.shape)

    return y,inputs

def der_tan_h(x):
    return (4/((e**x)+(e**(-x)))**2)


def phi(w,i):
    """
    Parameters
    ----------
    w : TYPE WEIGHTS
        DESCRIPTION.
    i : inputs  INPUTS
        DESCRIPTION.

    Returns
    -------
    function : TYPE
        DESCRIPTION.
    """
    
    function = w[0]*np.tanh(w[1]*i[0] + w[2]*i[1] + w[3]*i[2] + w[4] ) +w[5]*np.tanh(w[6]*i[0] + w[7]*i[1] + w[8]*i[2]+w[9]) + w[10]*np.tanh(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])+w[15]
    return function


def grad_func(w,i):
    """
    Parameters
    ----------
    w : TYPE WEIGHTS
        DESCRIPTION.
    i : inputs  INPUTS
        DESCRIPTION.

    Returns
    -------
    function : TYPE
        DESCRIPTION.
    """
    grad_function = np.array(np.tanh(w[1]*i[0] + w[2]*i[1] + w[3]*i[2]+w[4]))
    for j in i:
        grad_function = np.hstack(grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*j)
    grad_function = np.hstack(grad_function, w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4]))
    grad_function = np.hstack(grad_function,np.tanh(w[6]*i[0] + w[7]*i[1] + w[8]*i[2]+w[9]))
    for j in i:
        grad_function = np.hstack(grad_function,w[5]*der_tan_h(w[6]*i[0]+w[7]*i[1]+w[8]*i[2]+w[9])*j)
    grad_function = np.hstack(grad_function,w[5]*der_tan_h(w[6]*i[0]+w[7]*i[1]+w[8]*i[2]+w[9]))
    grad_function = np.hstack(grad_function,np.tanh(w[11]*i[0] + w[12]*i[1] + w[13]*i[2]+w[14]))
    for j in i:
        grad_function = np.hstack(grad_function,w[10]*der_tan_h(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])*j)
    grad_function = np.hstack(grad_function,w[10]*der_tan_h(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14]))
    grad_function = np.hstack(grad_function,1)
    print(grad_function)
    return grad_function

def jacobian_data(w,i):
    jacobian = np.empty((0,w.shape[0]))
    for j in i:
        jacobian = np.vstack(jacobian,grad_func(w,j))
    print(jacobian.shape)
    return jacobian

    
if __name__ == "__main__":
    
    L = 500
    
    Gamma = 1

    
    output, inputs = gen_data(Gamma,L)
    
    
    