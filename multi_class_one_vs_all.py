# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:00:30 2023

@author: jerem
"""
import scipy.io as sp
import scipy.linalg
import numpy as np
import pandas as pd
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
    #print(train_x.shape)
    #print(test_x.shape)
    one_vs_all_weights = np.empty((train_x.shape[1],0))
    for i in range(10):
        x= create_weights(train_x,if_matches(train_y,i))
        #print(x)
        
        one_vs_all_weights = np.concatenate((one_vs_all_weights,x),axis = 1) # this is all the weights for the different numbers 0-9
    
    preprocess_pred = test_x@one_vs_all_weights
    print(f"this is preprocess pred: {preprocess_pred}")
    prediction = np.argmax(preprocess_pred,axis = 1)
    print(f"this is prediction: {prediction}")
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
    print(train_x.shape)
    print("train_x shape")
    print(train_y.shape)
    print("train_y shape")
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
    list_of_weights = []
    counter = 0
    for i in range(9):
        for j in range(i+1,10):
            counter +=1
            (ovo_train_x,ovo_train_y) = one_vs_one_train_setup(train_x,train_y, (i,j))
            weights = create_weights(ovo_train_x,if_matches(ovo_train_y,i))

            list_of_weights.append(weights)
    weights = np.array(list_of_weights)
    #print(weights)
    #print(weights.shape)
    weights = np.squeeze(weights)

    tested_values = test_x@weights.transpose()
    #print(tested_values)

    tested_values_binary = np.sign(tested_values)
    print(tested_values_binary)
    tuple_columns = [(i, j) for i in range(9) for j in range(i+1, 10)]
    #print(tuple_columns)
    dataframe = pd.DataFrame(tested_values_binary, columns = tuple_columns)
    print(dataframe)

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
    #print(len(guess_value))
    final_guesses = np.array((guess_value))
    
    #print(final_guesses.shape)

    #print(final_guesses.shape)
    #print(final_guesses)
    #print(test_y)
    #guesses= counts.idxmax(axis=1)
    #guesses_int = guesses_str.astype(int)
    #print(guesses)
    return final_guesses


def analyze_multi_class(predicted, expected):
    columns = [f'Predicted {i}' for i in range(10)]
    indexes = [f'Expected {i}' for i in range(10)]
    print(f"this is expecteds shape : {expected.shape}")
    print(f"this is predicteds shape : {predicted.shape}")
    columns.append('Totals')
    indexes.append('Totals')
    confusion_multi_class = pd.DataFrame(data = None, columns = columns, index = indexes)
    confusion_multi_class.loc[:,:] = 0
    #print(predicted.shape)
    #print(expected.shape)
    for i in range(len(expected)):

        confusion_multi_class.loc[[f"Expected {expected[i][0]}"], [f"Predicted {predicted[i]}"]] += 1
    pd.set_option('display.max_columns', None)
    diagonal_sum = 1-(np.trace(confusion_multi_class.values)/len(predicted))
    confusion_multi_class.loc[["Totals"],['Totals']] = confusion_multi_class.sum().sum()
    for i in range(10):
        row_sum = confusion_multi_class.loc[f"Expected {i}"].sum()
        column_sum = confusion_multi_class[f"Predicted {i}"].sum()
        confusion_multi_class.loc[[f"Expected {i}"],['Totals']] = row_sum
        confusion_multi_class.loc[['Totals'],[f"Predicted {i}"]] = column_sum
    
    print(confusion_multi_class)
    print(f"Error is {diagonal_sum}")
    
    accuracy_df = pd.DataFrame(data = None, columns = ['Accuracy'], index = [f"Accuracy for {i}" for i in range(10)])
    accuracy_df.loc[:,:] = 0
    for i in range(10):
        denominator = confusion_multi_class.at[f"Expected {i}","Totals"]
        numerator = confusion_multi_class.at[f"Expected {i}",f'Predicted {i}']


        accuracy_df.loc[f"Accuracy for {i}"] = numerator/denominator
    print(accuracy_df)


def change_the_set(train_x,L,function_feature):
    print(f"trainx  shape[1] : {train_x.shape[1]}")
    W = np.random.normal(0, 1, (train_x.shape[1],L))
    b = np.random.normal(0, 1, (1,L))
    #print(f"train_x shape: {train_x.shape}") 
    #print(f"W shape: {W.shape}") 
    #print(f"b shape: {b.shape}") 
    new_train_x = train_x@W + b

    
    #setting up our features by running them through the chosen feature equation
    if function_feature == 1:
        new_train_x = new_train_x
    elif function_feature == 2:
        new_train_x = 1/(1+np.exp(-new_train_x))
    elif function_feature == 3:
        new_train_x = np.sin(new_train_x)
    elif function_feature == 4:
        new_train_x = np.maximum(new_train_x,0)
    print(f"new train_x shape: {new_train_x.shape}")       

    #print(f"new train_x shape: {new_train_x.shape}")
    return new_train_x

if __name__ == "__main__":
    file = "C:/Users/jerem/Downloads/mnist.mat"

    
    learning_dict = {}
    L = 1000 #feature mapping dimension
    function_feature = 2 #changing our function we pass through on our feature mapping (1-4)
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

    
    
    new_train_x = change_the_set(train_x,L,function_feature) #creating our new training set before we add our ones bias
    new_test_x = change_the_set(test_x,L,function_feature)
    
    new_test_ones = np.ones((new_test_x.shape[0],1))
    new_train_ones = np.ones((new_train_x.shape[0],1))
    
    new_train_x = np.append(new_train_x,new_train_ones,axis = 1)
    new_test_x = np.append(new_test_x,new_test_ones,axis = 1)
    
    
    train_x = np.append(train_x,ones, axis = 1)
    test_x = np.append(test_x,ones_test, axis = 1)
    
    cols_to_remove = np.all(train_x == 0, axis=0)



    train_x = train_x[:, ~cols_to_remove]

    test_x = test_x[:, ~cols_to_remove]

    
    
    
    #print(train_x.shape)
    
    
    print("\n")
    predictions_train_binary = binary_classifier(0,test_x,test_y,train_x,train_y)
    
    analyze_binary(predictions_train_binary,if_matches(test_y,0)) #running on test data now
    
    
    '''
    predicted_multi = one_vs_all_multi(train_x,train_y,test_x,test_y)
    analyze_multi_class(predicted_multi, test_y)
    #(ovo_train_x,ovo_train_y) = one_vs_one_train_setup(train_x,train_y, (1,2)) #returns the training data for the one vs one classifier
    
    #binary_ovo_predicted = one_vs_one_trainer(ovo_train_x,ovo_train_y,test_x,test_y, (1,2))
    '''
    """
    running our feature engineering on training data. 
    """
    '''
    
    predicted_featured_multi = one_vs_all_multi(new_train_x,train_y,new_train_x,train_y) #testing on the training data we add training data twice because the function signature asks for the data to test on seperately

    analyze_multi_class(predicted_featured_multi, train_y)# once again were running on training data ( I assume its going to be overfitting)

    #now lets try testing data
    '''
    '''
    running our new feature space on the testing data
    '''
    '''
    predicted_featured_multi_test = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y) #testing on the training data we add training data twice because the function signature asks for the data to test on seperately

    analyze_multi_class(predicted_featured_multi_test, test_y)
    
    '''
    
    
    #predicted_featured_multi_test = one_vs_all_multi(new_train_x,train_y,new_test_x,test_y)
    #print(f"this is  predicted_featured_multi_test shape : { predicted_featured_multi_test.shape}")
    #analyze_multi_class(predicted_featured_multi_test, test_y)
    
    """
    guesses = run_ovo_all(train_x,train_y,test_x,test_y)
    print(guesses.shape)
    analyze_multi_class(guesses, test_y)
    #analyze_binary(binary_ovo_predicted,if_matches(test_y,1))
    """