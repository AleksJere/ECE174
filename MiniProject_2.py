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
    
    #generates data from a normal distribution appends it to our inputs and outputs vector
    
    inputs = np.empty((0,3))
    outputs = np.empty((0,1))
    for i in range(n):
        data = np.random.uniform(-Gamma,Gamma,(3,))
        inputs = np.vstack((inputs,data))
        outputs = np.append(outputs,[data[0] * data[1] + data[2]])


    outputs = outputs.reshape(-1,1)
    #print(f"this is the shape of inputs {inputs.shape}")
    #print(f"this is the shape of outputs {outputs.shape}")
    return outputs,inputs

def gen_new_data(Gamma,n):
    inputs = np.empty((0,3))
    outputs = np.empty((0,1))
    for i in range(n):
        data = np.random.uniform(-Gamma,Gamma,(3,))
        inputs = np.vstack((inputs,data))
        outputs = np.append(outputs,[(data[0]**2)*max(data[1],1) + max(data[2],1)])
        
    outputs = outputs.reshape(-1,1)

    return outputs,inputs


def gen_data_with_noise(Gamma,n,eps):
    
    #generates data from a normal distribution appends it to our inputs and outputs vector
    
    inputs = np.empty((0,3))
    outputs = np.empty((0,1))
    for i in range(n):
        data = np.random.uniform(-Gamma,Gamma,(3,))
        inputs = np.vstack((inputs,data))
        outputs = np.append(outputs,[data[0] * data[1] + data[2] + np.random.uniform(-eps,eps)])


    outputs = outputs.reshape(-1,1)
    #print(f"this is the shape of inputs {inputs.shape}")
    #print(f"this is the shape of outputs {outputs.shape}")
    return outputs,inputs

def der_tan_h(x):
    #return (1- np.tanh(x)**2) #same thing was just making sure It wasnt an issue with how I show the derivative
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
    #print(f"this is the shape of i pre reshaping {i.shape}")
    
    #i = i.reshape(-1,1)
    #print(f"size of w in phi {w.shape} ")
    #print(f"size of i in phi {i.shape} ")
    
    
    #our function
    function = (w[0]*np.tanh(w[1]*i[0] + w[2]*i[1] + w[3]*i[2] + w[4] ) +w[5]*np.tanh(w[6]*i[0] + w[7]*i[1] + w[8]*i[2]+w[9]) + w[10]*np.tanh(w[11]*i[0] + w[12]*i[1] + w[13]*i[2] + w[14]) + w[15])
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
    
    #w = w.reshape(-1,1)
    #i = i.reshape(-1,1)
    #print(f"size of w {w.shape} ")
    #print(f"size of i {i.shape} ")
    
    #I have checked this many times this is definitely right and if it isnt i have no clue whats wrong with me but I really intensly made sure this should be right
    
    
    grad_function = np.array((np.tanh(w[1]*i[0] + w[2]*i[1] + w[3]*i[2]+w[4])))


    grad_function = np.hstack((grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*i[0]))
    grad_function = np.hstack((grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*i[1]))
    grad_function = np.hstack((grad_function,w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])*i[2]))
    
    
    grad_function = np.hstack((grad_function, w[0]*der_tan_h(w[1]*i[0]+w[2]*i[1]+w[3]*i[2]+w[4])))
    
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
    
    #creaitng our jacobian of size 500,16
    jacobian = np.empty((0,w.shape[0]))
    for j in range(len(i)):
        #print(i.shape)
        #print(f'shape of j {i[j].shape}')
        #print(f'shape of w {w.shape}')
        #print(f'shape of jacobian {jacobian.shape}')
        
        #print(f'shape of grad func {grad_func(lam,w,i[j]).shape}')
        jacobian = np.vstack((jacobian,grad_func(lam,w,i[j])))
    #print(f"jacobian shape = {jacobian.shape}")
    

    return jacobian

def phi_vector(w,i):
    
    #create our function as a vector of size 500,1
    full_vector = np.empty((0,1))
    #print(f"inputs has a shape of {i.shape}")
    for j in i:
        #print(f"j vector shape = {j.shape}")

        
        new_phi = np.asarray(phi(w,j))

        #print(f"new phi shape = {new_phi.shape}")
        full_vector = np.vstack((full_vector,new_phi))
    #print(f"full vector shape = {full_vector.shape}")

    return full_vector


def levenberg(outputs,inputs,weights = None, iteration_limit = 300,stop_value = .01,rho = 1,lam = 0.00005, std_dev = .1):
    
    #lambda is in function definition, as is my first step size rho

    if weights == None:
        weights = np.random.normal(0,std_dev, (16,1))
    #print(f"this is the shape of inputs in levenberg {inputs.shape}")
    
    #error calculation
    residual = np.sum((phi_vector(weights, inputs) - outputs)**2)
    error = residual + lam*(np.linalg.norm(weights))**2
    
    
    steps = [rho]
    
    errors = [error]


    
    trust = 0.8
    
    non_trust = 2
    

    while error > stop_value and len(errors) < iteration_limit and rho <1000: #for bad initilizations sometimes the rho can bounce to extremely high values and we would prefer if it didnt because its not useful after that point
        
        jacobian = jacobian_data(lam,weights,inputs)


        prediction = phi_vector(weights,inputs)

        #below this is the b matrix in discussion 9 

        y = jacobian@weights - (prediction - outputs)
        y_new = np.vstack((y,np.zeros((weights.shape)))) #accounts for the fact we will be adding the derivative of the lambda term from the loss function into the A matrix so we need this I think, I dont fully understand it but it wont work wiTHout it
        y_new = np.vstack((y_new, math.sqrt(rho)*weights))
        #print(f'this is rho {rho}')
        
        #this is our A matrix we will be using to calculate the linearized least squares we add our lambda from the loss function and the rho from levenberg
        A = np.vstack((jacobian, math.sqrt(lam)*np.identity(jacobian.shape[1])))
        A = np.vstack((A, math.sqrt(rho)*np.identity(jacobian.shape[1])))

        new_weights = np.linalg.pinv(A)@y_new #(A@(np.linalg.inv(A.T@A))@A.T)@y_new  #weights - np.linalg.inv(jacobian.T@jacobian+rho*np.identity(jacobian.shape[1]))@jacobian.T@y     #(A@(np.linalg.inv(A.T@A))@A.T)@y
        
        #error calculation
        residual = np.sum((phi_vector(new_weights,inputs) - outputs)**2)

        new_error = residual +lam*(np.linalg.norm(new_weights))**2

        #print(f"rho is {rho}")
        #print(f"error is {error}")
        #print(f"new error = {new_error}")

        if new_error < error:

            weights = new_weights
            rho = rho*trust
            error = new_error
            #print("trust")
        else:
            rho = rho*non_trust
            #print("no trust")
        steps.append(rho)
        errors.append(error)
        
        
        
    return errors, steps, weights




    
if __name__ == "__main__":
    
    L = 500
    
    Gamma = 1
    
    fig, arr = plt.subplots(2,1)
    fig.set_size_inches(10, 10)
    
    lines = []
    labels = []
    for i in range(1,5):
        for j in range(0,101,20):
            outputs, inputs = gen_data(Gamma,L)
            #print(f"this is the shape of inputs before levenberg {inputs.shape}")
            errors, steps, weights = levenberg(outputs,inputs,iteration_limit = 300,lam = 0.00005*j, stop_value = .1,rho = 1,std_dev = i*.1)
            

            
            print(errors[-1])
            
            line1, = arr[0].plot(errors, label=f'error intitilization standard_dev = {i*0.05} lambda = {0.00005*j}')
            arr[0].set_title(r'Training Loss over iterations')
            arr[0].set_xlabel('number of iterations')
            arr[0].set_ylabel('Training Loss')
            line2, = arr[1].plot(steps,label=f'steps intitilization standard_dev = {i*0.5} lambda = {0.00005*j}')
            arr[1].set_title(r'$\rho$ over iterations')
            arr[1].set_xlabel('number of iterations')
            arr[1].set_ylabel(r'$\rho$')
            labels.extend([f'error intitilization standard_dev = {i*0.05} lambda = {0.00005*j}', f'steps intitilization standard_dev = {i*0.05} lambda = {0.00005*j}'])
            
            lines.extend([line1, line2])

        
            #print(weights)
            
    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(lines, labels, 'center')
    figlegend.tight_layout()    
    plt.tight_layout()
    figlegend.savefig('legend_training_error_part_a.png')
    plt.show()
    
    
    
    
    Gamma = 1
    L = 500

    fig, arr = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    lines = []
    labels = []
    print('part 2')
    for i in range(1, 102, 20):  # lambda changing
        final_train_error = []
        final_test_error = []
        gamma_array = []
        
        for j in range(1, 22, 2):  # changing gamma
            lambda_value = 0.00005 * i
            gamma_value = (Gamma / 5) * j
            gamma_array.append(gamma_value)
    
            outputs, inputs = gen_data(gamma_value, L)
            errors, steps, weights = levenberg(outputs, inputs, iteration_limit=300, lam=lambda_value, stop_value=.3, rho=1, std_dev=.1)

            
            new_outputs, new_inputs = gen_data(gamma_value, 100)
            residual = np.sum((phi_vector(weights, new_inputs) - new_outputs) ** 2)
            error = residual + lambda_value * np.linalg.norm(weights)
    
            final_train_error.append(errors[-1])
            final_test_error.append(error)
    
        line1, = arr[0].plot(gamma_array, final_train_error, label=f'training lambda value = {lambda_value}')
        line2, = arr[1].plot(gamma_array, final_test_error, label=f'testing lambda value = {lambda_value}')
        
        lines.extend([line1, line2])
        labels.extend([f'Training, Lambda: {lambda_value}', f'Testing, Lambda: {lambda_value}'])

    arr[0].set_title('Final Training Loss by Gamma part_b')
    arr[0].set_xlabel('Gamma')
    arr[0].set_ylabel('Final Training Loss')

    arr[1].set_title('Final Testing Loss by Gamma')
    arr[1].set_xlabel('Gamma')
    arr[1].set_ylabel('Final Testing Loss')

    plt.tight_layout()

    # Create a separate figure for the legend
    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(lines, labels, loc='center')
    figlegend.tight_layout()
    figlegend.savefig('legend_changing_gamma_lambda_testing_training_part_b.png')

    # Save main figure
    plt.show()
    
    '''
    
    #New nonlinear function
    
    '''
    '''
    '''
    print('part 3 a')
    L = 500
    
    Gamma = 1

    fig, arr = plt.subplots(2,1)
    fig.set_size_inches(10, 10)

    lines = []
    labels = []
    for i in range(0,5):
        for j in range(1,102,20):
            outputs, inputs = gen_new_data(Gamma,L)
            #print(f"this is the shape of inputs before levenberg {inputs.shape}")
            errors, steps, weights = levenberg(outputs,inputs,iteration_limit = 300,lam = 0.00005*j, stop_value = .1,rho = 1,std_dev = i*0.1)
    
        
            line1, = arr[0].plot(errors, label=f'error intitilization standard_dev = {i*0.5} lambda = {0.00005*j}')
            arr[0].set_title(r'Training Loss over iterations')
            arr[0].set_xlabel('number of iterations')
            arr[0].set_ylabel('Training Loss')
            line2, = arr[1].plot(steps,label=f'steps intitilization standard_dev = {i*0.5} lambda = {0.00005*j}')
            arr[1].set_title(r'$\rho$ over iterations')
            arr[1].set_xlabel('number of iterations')
            arr[1].set_ylabel(r'$\rho$')
            labels.extend([f'error intitilization standard_dev = {i*0.5} lambda = {0.00005*j}', f'steps intitilization standard_dev = {i*0.5} lambda = {0.00005*j}'])
            
            lines.extend([line1, line2])
            
            
            
            #print(weights)
            
    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(lines, labels, 'center')
    figlegend.tight_layout()    
    plt.tight_layout()
    figlegend.savefig('legend_new_non_linear_part_a.png')
    plt.show()
    
    
    print('part 3 b')
    Gamma = 1
    L = 500
    
    fig, arr = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    lines = []
    labels = []
    
    for i in range(1, 102, 20):  # lambda changing
        final_train_error = []
        final_test_error = []
        gamma_array = []
        
        for j in range(1, 22, 2):  # changing gamma
            lambda_value = 0.00005 * i
            gamma_value = (Gamma / 5) * j
            gamma_array.append(gamma_value)
    
            outputs, inputs = gen_new_data(gamma_value, L)
            errors, steps, weights = levenberg(outputs, inputs, iteration_limit=300, lam=lambda_value, stop_value=.1, rho=1, std_dev=.1)
            
            new_outputs, new_inputs = gen_new_data(gamma_value, 100)
            residual = np.sum((phi_vector(weights, new_inputs) - new_outputs) ** 2)
            error = residual + lambda_value * (np.linalg.norm(weights))**2
    
            final_train_error.append(errors[-1])
            final_test_error.append(error)
    
        line1, = arr[0].plot(gamma_array, final_train_error, label=f'training lambda value = {lambda_value}')
        line2, = arr[1].plot(gamma_array, final_test_error, label=f'testing lambda value = {lambda_value}')
        
        lines.extend([line1, line2])
        labels.extend([f'Training, Lambda: {lambda_value}', f'Testing, Lambda: {lambda_value}'])

    arr[0].set_title('Final Training Loss by Gamma new function')
    arr[0].set_xlabel('Gamma')
    arr[0].set_ylabel('Final Training Loss')

    arr[1].set_title('Final Testing Loss by Gamma new function')
    arr[1].set_xlabel('Gamma')
    arr[1].set_ylabel('Final Testing Loss')

    plt.tight_layout()

    # Create a separate figure for the legend
    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(lines, labels, loc='center')
    figlegend.tight_layout()
    figlegend.savefig('legend_changing_gamma_lambda_new_data_part_c_part_b.png')

    # Save main figure
    plt.show()

    
    #part a and b again but with noise now
    
    
    print('part 4 a')
    
    L = 500
    
    Gamma = 1

    fig, arr = plt.subplots(2,1)
    fig.set_size_inches(10, 10)

    lines = []
    labels = []
    eps = .00001
    for i in range(0,5):
        for j in range(1,1001,200):
            outputs, inputs = gen_data_with_noise(Gamma,L,eps*j)
            #print(f"this is the shape of inputs before levenberg {inputs.shape}")
            errors, steps, weights = levenberg(outputs,inputs,iteration_limit = 300,lam = 0.00005, stop_value = .1,rho = 1,std_dev = 0.1*i)
        
            
            line1, = arr[0].plot(errors, label=f'error intitilization standard_dev = {i*0.5} lambda = {0.00005*j}')
            arr[0].set_title(r'Training Loss over iterations with noise')
            arr[0].set_xlabel('number of iterations')
            arr[0].set_ylabel('Training Loss')
            line2, = arr[1].plot(steps,label=f'steps intitilization standard_dev = {i*0.5} lambda = {0.00005*j}')
            arr[1].set_title(r'$\rho$ over iterations with noise')
            arr[1].set_xlabel('number of iterations')
            arr[1].set_ylabel(r'$\rho$')
            labels.extend([f'error intitilization standard_dev = {i*0.5} lambda = {0.00005*j}', f'steps intitilization standard_dev = {i*0.5} lambda = {0.00005*j}'])
                
            lines.extend([line1, line2])
            
            
            
            #print(weights)
            
    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(lines, labels, 'center')
    figlegend.tight_layout()    
    plt.tight_layout()
    figlegend.savefig('legend_part_a_with_noise_real_part_d.png')
    plt.show()
    
    
    print('part 4 b')
    
    Gamma = 1
    L = 500
    eps = .00001
    fig, arr = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    lines = []
    labels = []
    for z in range(1,1001,200):
        for i in range(1, 102, 20):  # lambda changing
            final_train_error = []
            final_test_error = []
            gamma_array = []
            
            for j in range(1, 22, 2):  # changing gamma
                lambda_value = 0.00005 * i
                gamma_value = (Gamma / 5) * j
                gamma_array.append(gamma_value)
        
                outputs, inputs = gen_data_with_noise(gamma_value, L,eps*z)
                errors, steps, weights = levenberg(outputs, inputs, iteration_limit=300, lam=lambda_value, stop_value=.1, rho=1, std_dev=.1)
                
                new_outputs, new_inputs = gen_data_with_noise(gamma_value, 100,eps*z)
                residual = np.sum((phi_vector(weights, new_inputs) - new_outputs) ** 2)
                error = residual + lambda_value * (np.linalg.norm(weights))**2
        
                final_train_error.append(errors[-1])
                final_test_error.append(error)
        
            line1, = arr[0].plot(gamma_array, final_train_error, label=f'training lambda value = {lambda_value} noise = {eps*z}')
            line2, = arr[1].plot(gamma_array, final_test_error, label=f'testing lambda value = {lambda_value} noise = {eps*z}')
            
            lines.extend([line1, line2])
            labels.extend([f'Training, Lambda: {lambda_value} noise = {eps*z}', f'Testing, Lambda: {lambda_value} noise = {eps*z}'])

    arr[0].set_title('Final Training Loss by Gamma with noise')
    arr[0].set_xlabel('Gamma')
    arr[0].set_ylabel('Final Training Loss')

    arr[1].set_title('Final Testing Loss by Gamma with noise')
    arr[1].set_xlabel('Gamma')
    arr[1].set_ylabel('Final Testing Loss')

    plt.tight_layout()

    # Create a separate figure for the legend
    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(lines, labels, loc='center')
    figlegend.tight_layout()
    figlegend.savefig('legend_changing_gamma_lambda_with_noise_in_legend.png')

    # Save main figure
    plt.show()
    


    

    