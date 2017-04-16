from os import listdir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
import pickle


def main():

    #Generating folder path information
    root_path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(root_path,'data','train.csv')
    test_path = os.path.join(root_path,'data','kaggle_test_data.csv')

    
    
    #Reading data from csv
    train_data = pd.read_csv(train_path)

    #Deleting attributes/features not necessary for analysis
    train = train_data.drop(['id','salary','education'],axis =1)
    #Parsing Classified output from Training data
    y_train = train_data['salary']

    #The part below divides the features into categorical and numeric.
    
    #Categorical data is one-hot encoded
    train_categorical = train.select_dtypes(include=['object']).copy()
    #print(train_object)
    train_categorical.head()
    train_categorical["sex"]= np.where(train_categorical["sex"].str.contains("Male"), 1, 0)
    #print(train_object)
    train_categorical = pd.get_dummies(train_categorical, columns=["workclass","marital-status","occupation","relationship","race","native-country"])

    #Numeric features parsed out of feature vector
    train_integer = train.select_dtypes(include=['int64']).copy()

    #Normalizing the numeric data
    
    train_integer['age'] = normalize_data(train_integer['age'])
    train_integer['fnlwgt'] = normalize_data(train_integer['fnlwgt'])
    train_integer['education-num'] = normalize_data(train_integer['education-num'])
    train_integer['capital-gain'] = normalize_data(train_integer['capital-gain'])
    train_integer['capital-loss'] = normalize_data(train_integer['capital-loss'])
    train_integer['hours-per-week'] = normalize_data(train_integer['hours-per-week'])
    train_new = np.concatenate((train_categorical, train_integer), axis=1)
    #print(np.shape(train_new))
    #phi_train = train_data[1:30000,1:89]
    #y_train = train_data[1:30000,89]
    #phi_test = train_data[30000:30010,1:89]


    #Reading Test data from csv
    test_data = pd.read_csv(test_path)

    

    #Cleansing Categorical attributes of Test data
    test = test_data.drop(['id','education'],axis = 1)
    test_categorical = test.select_dtypes(include=['object']).copy()
    test_categorical.head()
    test_categorical["sex"]= np.where(test_categorical["sex"].str.contains("Male"), 1, 0)
    test_categorical = pd.get_dummies(test_categorical, columns=["workclass","marital-status","occupation","relationship","race","native-country"])
    test_integer = test.select_dtypes(include=['int64']).copy()

    #Normalizing numerical attributes of test dataset
    
    test_integer['age'] = normalize_data(test_integer['age'])
    test_integer['fnlwgt'] = normalize_data(test_integer['fnlwgt'])
    test_integer['education-num'] = normalize_data(test_integer['education-num'])
    test_integer['capital-gain'] = normalize_data(test_integer['capital-gain'])
    test_integer['capital-loss'] = normalize_data(test_integer['capital-loss'])
    test_integer['hours-per-week'] = normalize_data(test_integer['hours-per-week'])
    test_new = np.concatenate((test_categorical, test_integer), axis=1)
    #print(np.shape(test_new))
    #y_test = train_data[30000:30010,89]

    #Number of Epochs
    num_passes = 2000
    print("No. of iterations = ",num_passes)


#Principal Component Analysis for Feature visualization and dimensionality reduction
    #Mean Vectors
    mean_train = calculate_mean(train_new)
    scatter_train = scatter(train_new,mean_train)
    

    #Eigen Value and Eigen Vector
    eig_value_train,eig_vect_train = np.linalg.eig(scatter_train)
    #show_eig(eig_value_train,eig_vect_train,train_new)
    #show_eig(eig_value_test,eig_vect_test,test_new)

    #Eigen Vectors
    visualize_eig(eig_value_train)
    

    #Feature data after selecting 43 principal components
    feature_train = feature_data(eig_value_train,eig_vect_train)
    

    #Transform to get new subspace
    train_final = train_new.dot(feature_train.T)
    

    #Build an ANN model
    model = build_NN(50,train_final,y_train,num_passes,True)
    writeDict(model,"weights.txt")#Writing weight vector to txt file
    print("Weight matrix has been stored as weight.txt.Please start classification")
    #model_test = readDict("/home/bodhisattwa/Desktop/ML Assignments/Assignment 2/weights.txt")
    #y_pred = predict(model_test,test_final)
    #print('Predicted class :',y_pred)
    
    
    
#Showing the eigen value,eigen vector pairs    
def feature_data(eig_value,eig_vect):
    eigen_pairs = [(np.abs(eig_value[i]),eig_vect[i,:]) for i in range(len(eig_value))]
    eigen_pairs.sort(key = lambda x:x[0],reverse = True)

    #for i in eigen_pairs:
        #print(i[0])
       
    feature_data = []
    for i in range(0,43):
        feature_data.append(eigen_pairs[i][1])
    feature_data = np.array(feature_data)
    return feature_data



#Plot to visualize eigen values
def visualize_eig(eig_value):
    fig = plt.figure(figsize=(10,5))
    sing_vals = np.arange(len(eig_value))+1
    plt.plot(sing_vals,eig_value,linewidth =2)
    plt.xlabel('PC')
    plt.ylabel('Eigen Value')
    plt.ylim(ymax = max(eig_value), ymin = min(eig_value))
    plt.title("Plot")
    plt.show()



#Generating Covariance matrix
def scatter(phi,mean_vector):
    scatter_matrix = np.zeros((np.shape(phi)[1],np.shape(phi)[1]))
    for i in range(np.shape(phi)[0]):
        scatter_matrix += (phi[i,:].reshape(phi.shape[1],1)-mean_vector).dot((phi[i,:].reshape(phi.shape[1],1)-mean_vector).T)
    return scatter_matrix


#Calculating feature mean vectors
def calculate_mean(phi):
    mean_vector = np.zeros(np.shape(phi))
    mean_vector = np.mean(phi,axis=0)
    return mean_vector
    
    
#Normalizing numerical features
def normalize_data(data):
    mean = np.mean(data)
    stdev = np.std(data)
    normalized_data = np.divide(np.subtract(data,mean),stdev)
    return normalized_data


#List of activation functions and their derivatives
def sigmoid(x):
    return 1/(1+exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
                       
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x)**2
                       

#Driver function to implement ANN
def build_NN(nn_hdim,phi,y,num_passes,print_loss =  False):

    num_examples = np.shape(phi)[0] #No. of training samples
    nn_input_dim = np.shape(phi)[1] #No. of features to be considered for training
    nn_output_dim = 2 #No. of classes
    #print("No. of training sample :",num_examples)
    #print("Input Dimension :",nn_input_dim)
    #print("Shape :",np.shape(phi))
    #Gradient descent Parameters

    learning_rate = 0.0001
    reg_lambda = 0.05 #Shrinkage Parameter
    
    #Initializing the parameters with random values.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim)
    b1 = np.zeros((1,nn_hdim))
    W2 = np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim)
    b2 = np.zeros((1,nn_output_dim))

    #Model would finally carry the weight vector and bias
    model = {}

    #Gradient Descent for every batch
    for i in range(0,num_passes):

        #Forward Propagation
        z1 = phi.dot(W1)+b1
        a1 = tanh(z1) #We could also try sigmoid/ReLU
        z2 = a1.dot(W2)+b2
        exp_scores = np.exp(z2)
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)

        #Implementing Backpropagation
        delta3 = probs
        #print(np.shape(probs))
        #print(np.shape(y))
        #print(probs)
        #print(y)
        #print(delta3[)
        delta3[range(0,num_examples),y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3,axis=0,keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1,2))
        dW1 = phi.T.dot(delta2)
        db1 = np.sum(delta2,axis=0)

        #Regularization Terms
        dW2 = dW2 + (reg_lambda * W2)
        dW1 = dW1 + (reg_lambda * W1)

        #Gradient Descent Parameter update
        W1 = W1 + (-learning_rate*dW1)
        b1 = b1 + (-learning_rate*db1)
        W2 = W2 + (-learning_rate*dW2)
        b2 = b2 + (-learning_rate*db2)

        #Assign new parameters to the model
        model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        

        #Printing the loss after every 100 iterations
        if print_loss and i%100 == 0:
            print("Loss after iteration %i: %f" %(i,calculate_loss(model,phi,y)))

    return model


#Function to predict an output
def predict(model,phi):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    #Forward propagation
    print(np.shape(phi))
    print(np.shape(W1))
    z1 = phi.dot(W1)+b1
    a1 = tanh(z1)
    z2 = a1.dot(W2)+b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) #Softmax function at output layer
    return np.argmax(probs,axis=1)



#Calculating Cross entropy loss after certain number of iterations
def calculate_loss(model,phi,y):

    learning_rate = 0.0001
    reg_lambda = 0.001 #Shrinkage Parameter
    num_examples = np.shape(phi)[0] #Number of Training Samples
    
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    #Forward propagation
    z1 = phi.dot(W1)+b1
    a1 = tanh(z1)
    z2 = a1.dot(W2)+b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims = True)
    #Calculating cross entropy loss
    cross_ent_loss = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(cross_ent_loss)
    #Adding regularization
    data_loss = data_loss + reg_lambda/2 * (np.sum(np.square(W1))+ np.sum(np.square(W2)))
    return 1./num_examples * data_loss 
    


# Save the weight vector generated by training as a txt file to be used later for testing
def writeDict(dict,filename):
    with open(filename,"wb") as handle:
        pickle.dump(dict,handle)
       
        

                                       
    


if __name__ == '__main__':
    main()
    
