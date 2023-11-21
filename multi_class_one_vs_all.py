# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:00:30 2023

@author: jerem
"""
import scipy.io as sp
import scipy.linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def create_weights(train_x,processed_train_y):
    """

    Parameters
    ----------
    train_x : TYPE np.array
        DESCRIPTION.
        
        this is our training set we will use this in conjunction with the normal equations (pseudoinverse) 
        
    processed_train_y : TYPE
        DESCRIPTION.

        this is our expected values (0-9) except we have alreafy run these expected values through a processor which sets
        1 -> to the value we are looking for
        -1 -> to any other digit        

    Returns
    -------
    pre_signed : TYPE
        DESCRIPTION.
        
        these are our weights matrix times our y, this gives us the final solution to the normal equation

    """

    solution = scipy.linalg.pinv(train_x)

    pre_signed = solution@processed_train_y #gives a matrix that is our weights 

    return pre_signed

def if_matches(testing,expected):
    """
    Parameters
    ----------
    testing : TYPE np.array
        DESCRIPTION.
        
        this is what our array is which we want to change to 1 or -1 based on what our expected value is, 
        
    expected : TYPE integer
        DESCRIPTION.
        
        The integer which we are going to use in relation to setting our y = 1 for the digit we want to find and y=-1 for the
        other digits
        
    Returns
    -------
    prediction_train_conversion : TYPE np.array
        DESCRIPTION.
        
        This is our new y (our training or expected results (the real digit value of our handwritten digits)) 
        that has been processed to -1 and 1 for the given digit
        
        
    """
    prediction_train_conversion = np.where(testing == expected,1,-1)
    return prediction_train_conversion 
def binary_classifier(digit_to_predict,test_x,test_y,train_x,train_y):
    """
    Parameters
    ----------
    digit_to_predict : TYPE
        DESCRIPTION.
        this is the digit we are going to predict for one vs all classifier

    Returns
    -------
    predictions_train_binary : TYPE
        DESCRIPTION.
        this reutrns our prediction in binary (1 means it is our digit) (-1 means not our digit)
    """
    
    weights = create_weights(train_x,if_matches(train_y,digit_to_predict)) #using test data now

    predictions_on_train = test_x@weights

    predictions_train_binary = np.sign(predictions_on_train)   
    
    return predictions_train_binary

def analyze_binary(predicted_value,expected_value):
    """
    

    Parameters
    ----------
    predicted_value : TYPE numpy array
        DESCRIPTION.
        
        the values that my binary classifier predicted
        
    expected_value : TYPE numpy array
        DESCRIPTION.
        ANALYZE THE BINARY CLASSIFIER USING confusion matrix

    Returns
    -------
    None.

    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    expected_true = np.count_nonzero(expected_value == 1)
    expected_false = np.count_nonzero(expected_value == -1)
    for i in range(len(expected_value)): #iterate over the list while also taking value
        if predicted_value[i] == expected_value[i] and predicted_value[i] == 1:
            true_positive += 1
        elif predicted_value[i] == expected_value[i] and predicted_value[i] == -1:
            true_negative += 1
        elif predicted_value[i] != expected_value[i] and expected_value[i] == 1:
                false_negative += 1

        else:
            false_positive += 1
    data_confusion_matrix = [[true_positive,false_negative,true_positive+false_negative],
                             [false_positive,true_negative,false_positive+true_negative],
                             [true_positive+false_positive,false_negative+true_negative,false_negative+true_negative+true_positive+false_positive]]
    indexes = ["Expected True",'Expected False','All']
    columns = ["Predicted True","Predicted_False",'Total']
    
    confusion_matrix = pd.DataFrame(data_confusion_matrix,columns = columns,index = indexes)
    error_rate = (false_positive + false_negative) / len(expected_value)
    true_positive_rate = true_positive / expected_true
    false_positive_rate = false_positive / expected_false
    true_negative_rate = true_negative / expected_false
    precision = true_positive / (true_positive + false_positive)
    false_negative_rate = false_negative / expected_false
    data = [error_rate,true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate,precision] 
    rates = pd.DataFrame(data,columns = ["rates"], index = ["error_rate","true_positive_rate","false_positive_rate","true_negative_rate","False Negative","precision"]) 
    print(confusion_matrix)
    print("\n")
    print(rates)
    
def one_vs_all_multi(train_x,train_y,test_x,test_y):
    """
    Parameters
    ----------
    train_x : TYPE np.array
        DESCRIPTION.
        
        the training data, we will be using this data to train our linear classifiers of each digit
        
    train_y : TYPE np.array
        DESCRIPTION.
        
        the training datas corresponding correct digit value, if were at training point x_i the digit 
        associated with that is y_i
        
    test_x : TYPE np.array
        DESCRIPTION.
        
        our data set we will be testing using our linear classifier we trained
        
    test_y : TYPE np.array
        DESCRIPTION.
        
        not actually used I just included it to look nice

    Returns
    -------
    prediction : TYPE np.array
        DESCRIPTION.
        
        the predicted values from (0-9)
    """
    one_vs_all_weights = np.empty((train_x.shape[1],0))
    for i in range(10):
        x= create_weights(train_x,if_matches(train_y,i))

        
        one_vs_all_weights = np.concatenate((one_vs_all_weights,x),axis = 1) # this is all the weights for the different numbers 0-9
    
    preprocess_pred = test_x@one_vs_all_weights

    prediction = np.argmax(preprocess_pred,axis = 1)
    
    prediction = prediction.reshape(-1,1)

    return prediction

def one_vs_one_train_setup(train_x,train_y, tuple_to_not_remove):
    """
    

    Parameters
    ----------
    train_x : TYPE
        DESCRIPTION.
        takes in training dataset
    train_y : TYPE
        DESCRIPTION.
        takes in the training dataset values
    tuple_to_not_remove : TYPE
        DESCRIPTION.

    Returns
    -------
    one_one_train_x : TYPE
        DESCRIPTION.
    one_one_train_y : TYPE
        DESCRIPTION.

    """

    
    cols_to_keep_1 = np.all(train_y != tuple_to_not_remove[0], axis=1)
    cols_to_keep_2 = np.all(train_y != tuple_to_not_remove[1], axis=1)
    first_train_x = train_x[ ~cols_to_keep_1,:]
    first_train_y = train_y[ ~cols_to_keep_1,:]
    second_train_x = train_x[ ~cols_to_keep_2,:]
    second_train_y = train_y[ ~cols_to_keep_2,:]
    
    train_x = np.vstack((first_train_x,second_train_x))
    train_y = np.vstack((first_train_y,second_train_y))

    return (train_x, train_y)


def one_vs_one_trainer(train_x,train_y,test_x,test_y, tuple_we_test_for):
    """
    

    Parameters
    ----------
    test_x : TYPE
        DESCRIPTION.
        Our testing set we have gotten,
    test_y : TYPE
        DESCRIPTION.
    tuple_we_test_for : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    weights = create_weights(train_x,if_matches(train_y,tuple_we_test_for[0])) #gives the weights of our matrix, this is for the first value, the other value is automatically considered -1
    
    tested_values = test_x@weights
    
    tested_values_binary = np.sign(tested_values) #our tested values on the whole set, this is our one vs one classifier trying to determine what is a 1 and what is a -1
    
    #print(tested_values_binary)
    
    
    return tested_values_binary


def run_ovo_all(train_x,train_y,test_x,test_y):
    """
    Parameters
    ----------
    train_x : TYPE nparray
        DESCRIPTION.
    train_y : TYPE np array
        DESCRIPTION.
    test_x : TYPE np array
        DESCRIPTION.
    test_y : TYPE np array
        DESCRIPTION.
        we run the one vs all tecnique except only on 2 digits, then we stack them all
        horizontally and implement our counting scheme
    Returns
    -------
    final_guesses : TYPE
        DESCRIPTION.
        our guesses based on our 45 one vs one classifiers
    """
    list_of_weights = []
    counter = 0
    for i in range(9):
        for j in range(i+1,10):
            counter +=1
            (ovo_train_x,ovo_train_y) = one_vs_one_train_setup(train_x,train_y, (i,j))
            weights = create_weights(ovo_train_x,if_matches(ovo_train_y,i))

            list_of_weights.append(weights)
    weights = np.array(list_of_weights)
    weights = np.squeeze(weights)

    tested_values = test_x@(weights.transpose())
    tested_values_binary = np.sign(tested_values)
    tuple_columns = [(i, j) for i in range(9) for j in range(i+1, 10)]

    dataframe = pd.DataFrame(tested_values_binary, columns = tuple_columns)

    counts_dict = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}
   
    guess_value = []
   
    for index,rows in dataframe.iterrows():
        for col_tuple,value in rows.items():
            a,b = col_tuple
            if value == 1:
                counts_dict[a] += 1
            elif value == -1:
                counts_dict[b] += 1
        guess_value.append([max(counts_dict, key = counts_dict.get)])
        
        for key in counts_dict:
            counts_dict[key] = 0
            
    final_guesses = np.array((guess_value))
    
    return final_guesses


def analyze_multi_class(predicted, expected):
    '''
    Parameters
    ----------
    predicted : TYPE
        DESCRIPTION.
        our input predicted array
    expected : TYPE
        DESCRIPTION.
        the actual value of the digit
    Returns
    -------
    None.
    this code tabulates all the error the matricies and finds the accuracies of all the digits
    '''
    columns = [f'Predicted {i}' for i in range(10)]
    indexes = [f'Expected {i}' for i in range(10)]
    columns.append('Totals')
    indexes.append('Totals')
    confusion_multi_class = pd.DataFrame(data = None, columns = columns, index = indexes)
    confusion_multi_class.loc[:,:] = 0

    for i in range(len(expected)):
        
        confusion_multi_class.loc[[f"Expected {expected[i][0]}"], [f"Predicted {predicted[i][0]}"]] += 1
    pd.set_option('display.max_columns', None)
    diagonal_sum = 1-(np.trace(confusion_multi_class.values)/len(predicted))
    confusion_multi_class.loc[["Totals"],['Totals']] = confusion_multi_class.sum().sum()
    for i in range(10):
        row_sum = confusion_multi_class.loc[f"Expected {i}"].sum()
        column_sum = confusion_multi_class[f"Predicted {i}"].sum()
        confusion_multi_class.loc[[f"Expected {i}"],['Totals']] = row_sum
        confusion_multi_class.loc[['Totals'],[f"Predicted {i}"]] = column_sum
    print("\n")
    print(confusion_multi_class)
    print("\n")
    print(f"Error is {diagonal_sum}")
    print("\n")
    accuracy_df = pd.DataFrame(data = None, columns = ['Accuracy'], index = [f"Accuracy for {i}" for i in range(10)])
    accuracy_df.loc[:,:] = 0
    for i in range(10):
        denominator = confusion_multi_class.at[f"Expected {i}","Totals"]
        numerator = confusion_multi_class.at[f"Expected {i}",f'Predicted {i}']

        
        accuracy_df.loc[f"Accuracy for {i}"] = numerator/denominator
    print(accuracy_df)
    return diagonal_sum


def change_the_set(train_x,test_x,L,function_feature):
    '''
    

    Parameters
    ----------
    train_x : TYPE np array
        DESCRIPTION.
    test_x : TYPE np array
        DESCRIPTION.
    L : TYPE integer
        DESCRIPTION.
        This is the dimensionality of our feature space, we can reduce or
        increase the dimensionality
    function_feature : TYPE integer
        DESCRIPTION.
        (1-4) this changes which function we use when finalizing our feature space
    Returns
    -------
    new_train_x : TYPE np array
        DESCRIPTION.
        our new feature space training data
    new_test_x : TYPE np array
        DESCRIPTION.
        our new feature space testing data

    '''
    W = np.random.normal(0, 1, (train_x.shape[1],L))
    b = np.random.normal(0, 1, (1,L))
 
    new_train_x = train_x@W + b
    new_test_x = test_x@W + b

    if function_feature == 1:
        new_train_x = new_train_x
        new_test_x = new_test_x
    elif function_feature == 2:
        new_train_x = 1/(1+np.exp(-new_train_x))
        new_test_x = 1/(1+np.exp(-new_test_x))
    elif function_feature == 3:
        new_train_x = np.sin(np.pi/180*new_train_x)
        new_test_x = np.sin(np.pi/180*new_test_x)
    elif function_feature == 4:
        new_train_x = np.maximum(new_train_x,0)
        new_test_x = np.maximum(new_test_x,0)

    return (new_train_x,new_test_x)

if __name__ == "__main__":
    file = "C:/Users/jerem/Downloads/mnist.mat"

    
    learning_dict = {}
    L = 1000 #feature mapping dimension
    function_feature = 4 #changing our function we pass through on our feature mapping (1-4)
    #testX is the testing data set
    #testY is the testing expected real values
    #trainX is the training data set
    #trainY is the training expected values
    sp.loadmat(file, mdict=learning_dict, appendmat=False)
    test_x = learning_dict['testX']
    test_y = learning_dict['testY'].transpose() #to make a column vector
    train_x = learning_dict['trainX']
    train_y = learning_dict['trainY'].transpose() #to make it a column vector

    
    
    ones = np.ones((train_x.shape[0],1))
    ones_test = np.ones((test_x.shape[0],1))
    
    
    
    train_x = train_x.astype(float) #normalizing as float then /255 to normalize between 0 and 1

    train_x /= 255
    test_x = test_x.astype(float)
    test_x  /=255 #normalizing  as float then /255 to normalize between 0 and 1

    
    
    new_train_x,new_test_x = change_the_set(train_x,test_x,L,function_feature) #creating our new training set before we add our ones bias
    #new_test_x = change_the_set(test_x,L,function_feature)
    
    new_test_ones = np.ones((new_test_x.shape[0],1))
    new_train_ones = np.ones((new_train_x.shape[0],1))
    
    new_train_x = np.append(new_train_x,new_train_ones,axis = 1)

    new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

    
    train_x = np.append(train_x,ones, axis = 1)
    test_x = np.append(test_x,ones_test, axis = 1)
    
    cols_to_remove = np.all(train_x == 0, axis=0)



    train_x = train_x[:, ~cols_to_remove]

    test_x = test_x[:, ~cols_to_remove]

    

    
    
    print("\n")
    
    '''
    binary classifier
    '''
    '''
    predictions_train_binary_feature = binary_classifier(0,new_test_x,test_y,new_train_x,train_y)
    
    analyze_binary(predictions_train_binary_feature,if_matches(test_y,0)) #running on test data now
    '''
    
    '''
    multiclass classifier
    '''
    '''
    predicted_multi_train = one_vs_all_multi(train_x,train_y,train_x,train_y)
    analyze_multi_class(predicted_multi_train, train_y)
    
    predicted_multi_test = one_vs_all_multi(train_x,train_y,test_x,test_y)
    analyze_multi_class(predicted_multi_test, test_y)
    
    '''
    #(ovo_train_x,ovo_train_y) = one_vs_one_train_setup(train_x,train_y, (1,2)) #returns the training data for the one vs one classifier
    
    #binary_ovo_predicted = one_vs_one_trainer(ovo_train_x,ovo_train_y,test_x,test_y, (1,2))
    
    '''
    running our one vs one classifier on both training and testing data
    '''
    '''
    guesses_train = run_ovo_all(train_x,train_y,train_x,train_y)


    analyze_multi_class(guesses_train, train_y)
    
    guesses_test = run_ovo_all(train_x,train_y,test_x,test_y)


    analyze_multi_class(guesses_test, test_y)
    
    '''
    
    '''
    Here we implement our feature engineering on the one vs all classifier to test how the error changes
    '''
    '''
    
    predicted_multi_train_feature = one_vs_all_multi(new_train_x,train_y,new_train_x,train_y)
    analyze_multi_class(predicted_multi_train_feature, train_y)
    
    predicted_multi_test_feature = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y)
    analyze_multi_class(predicted_multi_test_feature, test_y)
    
    '''
    '''
    We are now implementing our feature engineering onto the one vs one classifier
    '''
    '''
    guesses_featured = run_ovo_all(new_train_x,train_y,new_train_x,train_y)


    analyze_multi_class(guesses_featured, train_y)
    
    guesses_featured = run_ovo_all(new_train_x,train_y,new_test_x,test_y)

    
    analyze_multi_class(guesses_featured, test_y)
    '''
    L_features = [100*i for i in range(1,50)]
    
    error_train = [0 for i in range(1,50)]
    
    error_test = [0 for i in range(1,50)]
    
    
    for index,value in enumerate(L_features):
        
        new_train_x,new_test_x = change_the_set(train_x,test_x,value,function_feature)
        
        
        new_test_ones = np.ones((new_test_x.shape[0],1))
        new_train_ones = np.ones((new_train_x.shape[0],1))
        
        new_train_x = np.append(new_train_x,new_train_ones,axis = 1)

        new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

        guesses_featured_train = run_ovo_all(new_train_x,train_y,new_train_x,train_y)

        error_train[index] = analyze_multi_class(guesses_featured_train, train_y)
        

        
        guesses_featured_test = run_ovo_all(new_train_x,train_y,new_test_x,test_y)

        error_test[index] = analyze_multi_class(guesses_featured_test, test_y)
        plt.subplot(2, 1, 1)
        plt.plot(L_features, error_train)
        plt.title("training data")
        plt.xlabel("Dimensions")
        plt.ylabel("Error")
        plt.subplot(2, 1, 2)
        plt.plot(L_features, error_test)
        plt.title("test data")
        plt.xlabel("Dimensions")
        plt.ylabel("Error")
        plt.show()
        
    '''
    plt.plot(L_features, error_train)
    
    plt.plot(L_features, error_test)
    '''
    
    #analyze multiclass now returns error rate aswell
    


    




