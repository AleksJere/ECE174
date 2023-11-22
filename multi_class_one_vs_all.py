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

    solution = scipy.linalg.pinv(train_x)  # computing the pseudo inverse to find the solution to the normal equations

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
    prediction_train_conversion = np.where(testing == expected,1,-1) # for all i in testing (an array) 
    #if it equals expected(a scalar) we append a 1 to our predictions array and if they arent equal we append -1
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
    
    weights = create_weights(train_x,if_matches(train_y,digit_to_predict)) #creates weights using the training set and the expected values which
    #have been converted to (1,-1)

    predictions_on_train = test_x@weights # applying the weights to a test set

    predictions_train_binary = np.sign(predictions_on_train)   #taking the sign and if its positive (+1) -> it is our digit 
    #(-1) -> it isnt our digit
    
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
    expected_true = np.count_nonzero(expected_value == 1) # number of 1's we expect
    expected_false = np.count_nonzero(expected_value == -1)#number of -1's we expect
    
    for i in range(len(expected_value)): #iterate over the list
        if predicted_value[i] == expected_value[i] and predicted_value[i] == 1: #check if true positive
            true_positive += 1
        elif predicted_value[i] == expected_value[i] and predicted_value[i] == -1:# check if true negative
            true_negative += 1
        elif predicted_value[i] != expected_value[i] and expected_value[i] == 1:#check if false negative
                false_negative += 1

        else: #only false positive is left
            false_positive += 1
            
    #creating our data to store in our confusion matrix
    data_confusion_matrix = [[true_positive,false_negative,true_positive+false_negative],
                             [false_positive,true_negative,false_positive+true_negative],
                             [true_positive+false_positive,false_negative+true_negative,false_negative+true_negative+true_positive+false_positive]]
    indexes = ["Expected True",'Expected False','All'] #indexes of our confusion matrix
    columns = ["Predicted True","Predicted_False",'Total'] #columns of our confusion matrix
    
    confusion_matrix = pd.DataFrame(data_confusion_matrix,columns = columns,index = indexes) #create the dataframe and put all the data in it
    error_rate = (false_positive + false_negative) / len(expected_value) #compute error rate
    true_positive_rate = true_positive / expected_true #compute true positive rate
    false_positive_rate = false_positive / expected_false #compute the false positive rate
    true_negative_rate = true_negative / expected_false #compute the true negative rate
    precision = true_positive / (true_positive + false_positive) #compute the precision (all of this is given in the book)
    false_negative_rate = false_negative / expected_false #compute false negative rate
    data = [error_rate,true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate,precision] #set our values as data
    rates = pd.DataFrame(data,columns = ["rates"], index = ["error_rate","true_positive_rate","false_positive_rate","true_negative_rate","False Negative","precision"]) 
    # creating our dataframe (couldve done a series too) of our rates
    print(confusion_matrix) #printing the confusion matrix
    print("\n")
    print(rates) #printing the error rates information
    
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
    one_vs_all_weights = np.empty((train_x.shape[1],0))  #creating an empty np array to hold our weights
    for i in range(10): #loop through all the one vs all digits we want to make weights for (0-9)
        x= create_weights(train_x,if_matches(train_y,i)) #create weights for the specific i value (0-9)

        
        one_vs_all_weights = np.concatenate((one_vs_all_weights,x),axis = 1) # this is all the weights for the different numbers 0-9 are concatenated (added to array horizontally)
    
    preprocess_pred = test_x@one_vs_all_weights # apply our weights to our testing data

    prediction = np.argmax(preprocess_pred,axis = 1) #see which of the digits (0-9) had the highest confidence (basically if the value is more positive ie .1 versus .5 we consider the .5 to be a more confident guess)
    
    prediction = prediction.reshape(-1,1) # I reshape the size of the predictions because it was of size (10000,) from the argmax procedure and when I do my one vs one it was of size (10000,1) 
    #so i wanted to reuse my code for all the claculations and tabulations so i just reshape it (doesnt affect anything else)
    return prediction #return our prediction array

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

    
    cols_to_keep_1 = np.all(train_y != tuple_to_not_remove[0], axis=1) #only keep the digits we are going to train this specific one vs one classifier on
    cols_to_keep_2 = np.all(train_y != tuple_to_not_remove[1], axis=1) #this is the second digit we want to keep (above is the first)
    first_train_x = train_x[ ~cols_to_keep_1,:] #keeps only the first digit we want to use
    first_train_y = train_y[ ~cols_to_keep_1,:]# same thing but for the expected values
    second_train_x = train_x[ ~cols_to_keep_2,:]#keeps only the second digit we want to use
    second_train_y = train_y[ ~cols_to_keep_2,:] # keep only the second digit but for the expected values
    
    train_x = np.vstack((first_train_x,second_train_x)) #stack the training vertically so its a new dataset of only 2 digits to classify between
    train_y = np.vstack((first_train_y,second_train_y)) #stacks the expected values the same way so order is preserved

    return (train_x, train_y) #return both


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
    list_of_weights = []  #initilizing our list of weights

    for i in range(9): #iterate over the first digit we are permutating through
        for j in range(i+1,10): #this for loop runs throught the second digit, the range(i+1,10) means that the code will never do something like (1,0) and will only increment where the second digit is greater than the first removing any repeats

            (ovo_train_x,ovo_train_y) = one_vs_one_train_setup(train_x,train_y, (i,j)) # set up the training data
            weights = create_weights(ovo_train_x,if_matches(ovo_train_y,i)) #create weights based on our newest one vs one classifier

            list_of_weights.append(weights) #append these weights
    weights = np.array(list_of_weights) #convert to np array
    weights = np.squeeze(weights) # remove the extra dimension that appears because we started with a python list it becomes a (x,y,z) array where one values is one so I remove it because it has no significance to the problem just an issue of converting lists to np arrays

    tested_values = test_x@(weights.transpose()) #we transpose the weights here because when we made our weights with the list, we werent stacking it correctly so the transpose fixes our dimension issue
    tested_values_binary = np.sign(tested_values) #take the sign value of our new predictions
    tuple_columns = [(i, j) for i in range(9) for j in range(i+1, 10)] # we are creating the columns of the pandas data frame which we will use to make our voting system

    dataframe = pd.DataFrame(tested_values_binary, columns = tuple_columns) #create dataframe with the (1,-1) predictions as the rows that correspond to each tuple (column)

    counts_dict = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0} #our counts dictionary
   
    guess_value = [] #create the list we will be appending our guess to
   
    for index,rows in dataframe.iterrows(): #run through all the rows of our dataframe
        for col_tuple,value in rows.items(): # run through every column and collect the column name (our tuple) and the value corresponding
            a,b = col_tuple #seperate our tuple so we know whats a +1 and whats a -1 (a +1 would mean a is our guess -1 means b is our guess , this is how I made the one vs one classifier)
            if value == 1: #if the value is one increment a for the above reason
                counts_dict[a] += 1
            elif value == -1: #if the value is -1 increment b for the same reason
                counts_dict[b] += 1
        guess_value.append([max(counts_dict, key = counts_dict.get)]) #append whatever digit got the most counts to our guess and thats now our guess for that digit
        
        for key in counts_dict: #reset the counts dictionary (Aside (I was wondering why my one vs one loved guessing the digit 8 turns out its cause I forgot to reset the counts))
            counts_dict[key] = 0
            
    final_guesses = np.array((guess_value)) # create a np array of our guesses
    
    return final_guesses #return our guesses


def analyze_multi_class(predicted, expected): #analyzingour multi class classifiers
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
    columns = [f'Predicted {i}' for i in range(10)] #create the columns of predicted values
    indexes = [f'Expected {i}' for i in range(10)] #create columns of expected values from our y 
    columns.append('Totals') #adding our totals column
    indexes.append('Totals') # adding our totals index
    confusion_multi_class = pd.DataFrame(data = 0, columns = columns, index = indexes) #creating our dataframe and initilizing the data to 0's

    for i in range(len(expected)): # for every single guess
        
        confusion_multi_class.loc[[f"Expected {expected[i][0]}"], [f"Predicted {predicted[i][0]}"]] += 1 #append one to the where the guess happend and where it was expected (I was proud of this piece of work)
    
    pd.set_option('display.max_columns', None) #display all of the columns so that when I screenshot them I can get all of them
    diagonal_sum = 1-(np.trace(confusion_multi_class.values)/len(predicted)) #this finds the error
    confusion_multi_class.loc[["Totals"],['Totals']] = confusion_multi_class.sum().sum() #append the total sum of elements to make sure were getting all 60000/10000
    for i in range(10): #iterate through all the rows/columns by using indexing
        row_sum = confusion_multi_class.loc[f"Expected {i}"].sum() #sum the rows
        column_sum = confusion_multi_class[f"Predicted {i}"].sum() #sum the columns
        confusion_multi_class.loc[[f"Expected {i}"],['Totals']] = row_sum #append the sum of the rows to the totals section of the ith expected row
        confusion_multi_class.loc[['Totals'],[f"Predicted {i}"]] = column_sum #append the sum of the columns to the totals section of the ith expected column
    print("\n")
    print(confusion_multi_class) #print our large tabulated data
    print("\n")
    print(f"Error is {diagonal_sum}") #print error
    print("\n")
    accuracy_df = pd.DataFrame(data = 0, columns = ['Accuracy'], index = [f"Accuracy for {i}" for i in range(10)]) #creating a dataframe of the accuracy of every digit
    for i in range(10): #iterate through every digit
        denominator = confusion_multi_class.at[f"Expected {i}","Totals"] # take total expected times to predict
        numerator = confusion_multi_class.at[f"Expected {i}",f'Predicted {i}'] #take the times we did predict correctly

        
        accuracy_df.loc[f"Accuracy for {i}"] = numerator/denominator #divide to find accuracy
    print(accuracy_df) #printing the accuracies of every digit
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
    W = np.random.normal(0, 1, (train_x.shape[1],L)) #create a random W matrix of size 784 (the size of our images) by L the dimension of the feature space we want to convert it to
    b = np.random.normal(0, 1, (1,L)) #create the b vector with the same dimension L of the same random distribution
 
    new_train_x = train_x@W + b #create our new feature space for the training data
    new_test_x = test_x@W + b #create the new feature space for the testing data

    if function_feature == 1: # changing the function which we put our images through this is identity function
        new_train_x = new_train_x
        new_test_x = new_test_x
    elif function_feature == 2: # sigmoid function
        new_train_x = 1/(1+np.exp(-new_train_x))
        new_test_x = 1/(1+np.exp(-new_test_x))
    elif function_feature == 3: # sinusoidal function which takes values in radians and our values after applying the random weights is in the range of dozens so we need to convert to +- pi region so we can actually retrieve information so we convert to radians
        new_train_x = np.sin(np.pi/180*new_train_x)
        new_test_x = np.sin(np.pi/180*new_test_x)
    elif function_feature == 4: #ReLU function
        new_train_x = np.maximum(new_train_x,0)
        new_test_x = np.maximum(new_test_x,0)

    return (new_train_x,new_test_x) #return our new set of training and testing vectors

if __name__ == "__main__":
    file = "C:/Users/jerem/Downloads/mnist.mat"

    
    learning_dict = {}
    L = 1000 #feature mapping dimension
    function_feature = 4 #changing our function we pass through on our feature mapping (1-4)
    #testX is the testing data set
    #testY is the testing expected real values
    #trainX is the training data set
    #trainY is the training expected values
    sp.loadmat(file, mdict=learning_dict, appendmat=False) #loading our data into python
    test_x = learning_dict['testX'] #makes it easier to type out only train_x
    test_y = learning_dict['testY'].transpose() #to make a column vector
    train_x = learning_dict['trainX']
    train_y = learning_dict['trainY'].transpose() #to make it a column vector

    
    
    ones = np.ones((train_x.shape[0],1)) #creating a ones vector which we will apply as our bias (alpha)
    ones_test = np.ones((test_x.shape[0],1)) #doing the same on our test data set
    
    
    
    train_x = train_x.astype(float) #normalizing as float then /255 to normalize between 0 and 1

    train_x /= 255
    test_x = test_x.astype(float)
    test_x  /=255 #normalizing  as float then /255 to normalize between 0 and 1

    
    
    new_train_x,new_test_x = change_the_set(train_x,test_x,L,function_feature) #creating our new training set before we add our ones bias this is for future use when we use our feature mapping
    #new_test_x = change_the_set(test_x,L,function_feature)
    
    new_test_ones = np.ones((new_test_x.shape[0],1)) #creating bias (alpha) for our feature mapped train_x and feature mapped test_x
    new_train_ones = np.ones((new_train_x.shape[0],1))
    
    new_train_x = np.append(new_train_x,new_train_ones,axis = 1) #appending our bias to the feature mapped train and test x 

    new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

    
    train_x = np.append(train_x,ones, axis = 1) #appending our bias to the regular (input space) data set
    test_x = np.append(test_x,ones_test, axis = 1)
    
    cols_to_remove = np.all(train_x == 0, axis=0) #finding the columns of our image that have no information (all 0's) we dont do this for the feature mapping because after the feature mapping there is no 0's columns anymore



    train_x = train_x[:, ~cols_to_remove] #removing the useless columns from the training and testing data set (important to remove the same from both)

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
    
    
    
    """
    
    L_features = [400*i+100 for i in range(7)]
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]
    '''
    one vs all  function feature identity
    '''
    function_feature = 1
    for index,value in enumerate(L_features):
        
        new_train_x,new_test_x = change_the_set(train_x,test_x,value,function_feature)
        
        
        new_test_ones = np.ones((new_test_x.shape[0],1))
        new_train_ones = np.ones((new_train_x.shape[0],1))
        
        new_train_x = np.append(new_train_x,new_train_ones,axis = 1)

        new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

        guesses_featured_train = one_vs_all_multi(new_train_x,train_y,new_train_x,train_y)

        error_train[index] = analyze_multi_class(guesses_featured_train, train_y)
        

        
        guesses_featured_test = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y)

        error_test[index] = analyze_multi_class(guesses_featured_test, test_y)
        
    plt.subplot(2, 1, 1)
    plt.plot(L_features, error_train)
    plt.title("training data identity ova")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
    
    
   
    '''
    one vs all multi function feature sigmoid
    '''
    
    function_feature = 2 
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]

    for index,value in enumerate(L_features):
        
        new_train_x,new_test_x = change_the_set(train_x,test_x,value,function_feature)
        
        
        new_test_ones = np.ones((new_test_x.shape[0],1))
        new_train_ones = np.ones((new_train_x.shape[0],1))
        
        new_train_x = np.append(new_train_x,new_train_ones,axis = 1)

        new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

        guesses_featured_train = one_vs_all_multi(new_train_x,train_y,new_train_x,train_y)

        error_train[index] = analyze_multi_class(guesses_featured_train, train_y)
        

        
        guesses_featured_test = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y)

        error_test[index] = analyze_multi_class(guesses_featured_test, test_y)
    
    plt.subplot(2, 1, 1)
    plt.plot(L_features, error_train)
    plt.title("training data ova sigmoid")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
    
    '''
    one vs all multi function feature sinusoidal
    '''
    
    function_feature = 3
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]

    for index,value in enumerate(L_features):
        
        new_train_x,new_test_x = change_the_set(train_x,test_x,value,function_feature)
        
        
        new_test_ones = np.ones((new_test_x.shape[0],1))
        new_train_ones = np.ones((new_train_x.shape[0],1))
        
        new_train_x = np.append(new_train_x,new_train_ones,axis = 1)

        new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

        guesses_featured_train = one_vs_all_multi(new_train_x,train_y,new_train_x,train_y)

        error_train[index] = analyze_multi_class(guesses_featured_train, train_y)
        

        
        guesses_featured_test = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y)

        error_test[index] = analyze_multi_class(guesses_featured_test, test_y)
    plt.subplot(2, 1, 1)
    plt.plot(L_features, error_train)
    plt.title("training data ova sinusoidal")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
        
        
        
    '''
    one vs all multi function feature ReLU
    '''
    
    function_feature = 4
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]

    for index,value in enumerate(L_features):
        
        new_train_x,new_test_x = change_the_set(train_x,test_x,value,function_feature)
        
        
        new_test_ones = np.ones((new_test_x.shape[0],1))
        new_train_ones = np.ones((new_train_x.shape[0],1))
        
        new_train_x = np.append(new_train_x,new_train_ones,axis = 1)

        new_test_x = np.append(new_test_x,new_test_ones,axis = 1)

        guesses_featured_train = one_vs_all_multi(new_train_x,train_y,new_train_x,train_y)

        error_train[index] = analyze_multi_class(guesses_featured_train, train_y)
        

        
        guesses_featured_test = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y)

        error_test[index] = analyze_multi_class(guesses_featured_test, test_y)
    
    plt.subplot(2, 1, 1)
    plt.plot(L_features, error_train)
    plt.title("training data ova ReLU")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
        
        
        
        
    '''
    one vs one feature function identity
    '''
        
    function_feature = 1
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]
    
    
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
    plt.title("training data ovo identity")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
    
    
    '''
    one vs one feature function sigmoid
    '''
        
    function_feature = 2
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]
    
    
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
    plt.title("training data ovo sigmoid")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
    
    
    '''
    one vs one feature function sinusoidal
    '''
        
    function_feature = 3
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]
    
    
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
    plt.title("training data ovo sinusoidal")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
    
    
    '''
    one vs one feature function ReLU
    '''
        
    function_feature = 4
    
    error_train = [0 for i in range(0,7)]
    
    error_test = [0 for i in range(0,7)]
    
    
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
    plt.title("training data ovo ReLU")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(L_features, error_test)
    plt.title("test data")
    plt.xlabel("Dimensions")
    plt.ylabel("Error")
    plt.show()
    
    """
    
    #here I am attempting to see if the sigmoid function ever overfits in the ovo case
    
    new_train_x,new_test_x = change_the_set(train_x,test_x,4000,2)
    
    guesses_featured = run_ovo_all(new_train_x,train_y,new_train_x,train_y)


    analyze_multi_class(guesses_featured, train_y)
    
    guesses_featured = run_ovo_all(new_train_x,train_y,new_test_x,test_y)

    
    analyze_multi_class(guesses_featured, test_y)
    
    
    
    
    
    '''
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
        
    '''
    plt.plot(L_features, error_train)
    
    plt.plot(L_features, error_test)
    '''
    
    #analyze multiclass now returns error rate aswell