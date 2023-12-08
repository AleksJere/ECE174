# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:08:49 2023

@author: jerem
"""

import numpy as np

import math
import matplotlib.pyplot as plt
e = np.e

def gen_data(Gamma,n):
    inputs = np.random.uniform(-Gamma,Gamma,(n,3))
    y =  inputs[:, 0] * inputs[:, 1] + inputs[:,2]
    y = y.reshape(n,1)

    #print(f"this is the shape of inputs {inputs.shape}")
    #print(f"this is the shape of y {y.shape}")
    return y,inputs

def der_tan_h(x):
    return (4/(((e**x)+(e**(-x)))**2))


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
    
    i = i.reshape(-1,1)
    #print(f"size of w in phi {w.shape} ")
    #print(f"size of i in phi {i.shape} ")
    
    function = (w[0]*np.tanh(w[1]*i[0] + w[2]*i[1] + w[3]*i[2] + w[4] ) +w[5]*np.tanh(w[6]*i[0] + w[7]*i[1] + w[8]*i[2]+w[9]) + w[10]*np.tanh(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])+w[15])
    return function


def grad_func(lam,w,i):
    """
    Parameters
    ----------
    w : TYPE WEIGHTS
        DESCRIPTION.
    i : inputs  INPUTS
        DESCRIPTION.
        math.sqrt(lam) : innput,  the value of math.sqrt(lam)ba for our loss function
    Returns
    -------
    function : TYPE
        DESCRIPTION.
    """
    
    w = w.reshape(-1,1)
    #i = i.reshape(-1,1)
    #print(f"size of w {w.shape} ")
    #print(f"size of i {i.shape} ")
    grad_function = np.array((np.tanh(w[1]*i[0] + w[2]*i[1] + w[3]*i[2]+w[4])))
    #grad_function = grad_function.reshape(-1,1)
    #print(f"shape of grad_function {grad_function.shape}")

    grad_function = np.hstack((grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*i[0]))
    grad_function = np.hstack((grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*i[1]))
    grad_function = np.hstack((grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*i[2]))
    
    
    grad_function = np.hstack((grad_function, w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2])))
    grad_function = np.hstack((grad_function,np.tanh(w[6]*i[0] + w[7]*i[1] + w[8]*i[2]+w[9])))
        
    grad_function = np.hstack((grad_function,w[5]*der_tan_h(w[6]*i[0]+w[7]*i[1]+w[8]*i[2]+w[9])*i[0]))
    grad_function = np.hstack((grad_function,w[5]*der_tan_h(w[6]*i[0]+w[7]*i[1]+w[8]*i[2]+w[9])*i[1]))
    grad_function = np.hstack((grad_function,w[5]*der_tan_h(w[6]*i[0]+w[7]*i[1]+w[8]*i[2]+w[9])*i[2]))    
        
    
    grad_function = np.hstack((grad_function,w[5]*der_tan_h(w[6]*i[0]+w[7]*i[1]+w[8]*i[2]+w[9])))
    grad_function = np.hstack((grad_function,np.tanh(w[11]*i[0] + w[12]*i[1] + w[13]*i[2]+w[14])))

    grad_function = np.hstack((grad_function,w[10]*der_tan_h(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])*i[0]))
    grad_function = np.hstack((grad_function,w[10]*der_tan_h(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])*i[1]))
    grad_function = np.hstack((grad_function,w[10]*der_tan_h(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])*i[2]))    
    
    
    grad_function = np.hstack((grad_function,w[10]*der_tan_h(w[11]*i[0]+w[12]*i[1]+w[13]*i[2]+w[14])))
    grad_function = np.hstack((grad_function,1))
    
    return grad_function

def jacobian_data(lam,w,i):
    jacobian = np.empty((0,w.shape[0]))
    for j in i:
        jacobian = np.vstack((jacobian,grad_func(lam,w,j)))
    #print(f"jacobian shape = {jacobian.shape}")
    return jacobian

def phi_vector(w,i):
    full_vector = np.empty((0,1))
    #print(f"inputs has a shape of {i.shape}")
    for j in i:
        #print(f"full vector shape = {full_vector.shape}")

        
        new_phi = np.asarray(phi(w,i))
        new_phi = new_phi.reshape(-1,1)
        #print(f"new phi shape = {new_phi.shape}")
        full_vector = np.vstack((full_vector,new_phi))
    #print(f"full vector shape = {full_vector.shape}")

    #print(f"new phi shape = {new_phi.shape}")
    return full_vector


def levenberg(outputs,inputs,weights = None, iteration_limit = 200,stop_value = 1,rho = 1,lam = 0.00005,random = True, std_dev = .5):
    
    if weights == None and random == False:
        weights = np.zeros((16,1))
    if random == True and weights == None:
        weights = np.random.normal(0,std_dev, (16,1))
    #print(f"this is the shape of inputs in levenberg {inputs.shape}")
    residual = np.sum((phi_vector(weights, inputs) - outputs)**2)
    error = residual + lam*np.linalg.norm(weights)
    
    rho = 1
    
    steps = [rho]
    
    errors = [error]


    
    trust = 0.8
    
    non_trust = 2
    

    while error > stop_value and len(errors) < iteration_limit:
        
        jacobian = jacobian_data(lam,weights,inputs)
        #print(f"size of inputs in levenberg before phi vecotr {inputs.shape}")
        prediction = phi_vector(weights,inputs)

        exp_values = jacobian@weights -outputs  
        
        exp_values = np.vstack((exp_values, math.sqrt(rho)*weights))
        
        
        
        new_A = np.vstack((jacobian, math.sqrt(rho)*np.identity(jacobian.shape[1])))


        new_weights = np.linalg.pinv(new_A)@exp_values
        
        residual = np.sum((phi_vector(new_weights,inputs) - outputs)**2)

        new_error = residual +lam*np.linalg.norm(new_weights)

        print(f"rho is {rho}")
        print(f"error is {error}")
        print(f"new error = {new_error}")

        if new_error < error:

            weights = new_weights
            rho = rho*trust
            error = new_error
            print("trust")
        else:
            rho = rho*non_trust
            print("no trust")
        steps.append(rho)
        errors.append(error)
        
        
        
    return errors, steps, weights




    
if __name__ == "__main__":
    
    L = 500
    
    Gamma = 1

    
    outputs, inputs = gen_data(Gamma,L)
    #print(f"this is the shape of inputs before levenberg {inputs.shape}")
    errors, steps, weights = levenberg(outputs,inputs,stop_value = .3,rho = 1)
    
    fig, arr = plt.subplots(2,1)
    fig.set_size_inches(10, 10)

    arr[0].plot(errors)
    arr[0].set_title(r'Training Loss over iterations')
    arr[0].set_xlabel('number of iterations')
    arr[0].set_ylabel('Training Loss')
    arr[1].plot(steps)
    arr[1].set_title(r'$\rho$ over iterations')
    arr[1].set_xlabel('number of iterations')
    arr[1].set_ylabel(r'$\rho$')
    print(weights)