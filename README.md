# NeuralNetworkFromScratch

This deep learning project is made alongside the "Neural Networks From Scratch" e-book. 
The e-book goes over the basics of neural networks, such as neurons, activation functions and loss functions. As demonstrated in the project,
these fundamentals have been coded from scratch. The book also cover intermediate techniques for neural networks such as stochastic gradient descent,
optimization functions, and regularization. 

The book also shows you how to load datasets, use them to train networks, save the networks, and how to load them again. 
All of this is done without the use of a deep learning framework. 

This project is intended to show the underlying computations in neural networks that are typically hidden when using deep learning frameworks. This
project was also a learning experience for me as this was my first hands-on experience with neural networks. 


# Table of Contents 

## NNLearning.py

This file contains all the code you would need for a basic neural network. 

### Layer
Creates a layer with the specified number of neurons. Takes in the the input of previous layer (or the actual input of the model) and computes the ouputs using the automatically initialized weights and biases. 

### dropOutReg
Perfoms regularization on a layer using Dropout. 

### ReLuActivation
Performs relu ativation on the output of the layer

### sigmoidActivation
Performs sigmoid ativation on the output of the layer

### linear_activation
Performs no activation on the outputs. This object exists as a placeholder for an activation object for optimization functions and gradient descent. 

### softMaxActivation
Performs the softmax activation on the output layer. 

### Loss
Parent object for the different loss functions. Calculates the mean of the loss across the input batch. 

### Loss_CategoricalCrossentropy
Calculates the loss of the input using Catergorical Cross Entropy.

### Loss_BinaryCrossentropy
Calculates the loss of the input using Binary Cross Entropy.

### momentumSGD
Uses the optimization function 'momentum' to speed up gradient descent and avoid local minima.

### adamOptimizer
Uses the optimization function 'adam' to speed up gradient descent and avoid local minima.

### activation_softmax_loss_categoricalCrossentropy
Combined object of softmax activation and binary cross entropy.

### loss_MeanSquaredError
Calculates the loss of the input by taking the mean of squared difference between the ground truth vector and the output. 

### Loss_MeanAbsoluteError
Calculates the loss of the input by taking the mean of the absolute difference between the ground truth vector and the output. 

### Model
An object that is used to construct a full neural network using the objects mentioned above, performs gradient descent on the network, and saves and loads neural networks. 

### Layer_Input:
This object is added to the start of a neural network, stores the input of the model and acts as a layer. 

### Accuracy
Parent class that calculates accuracy of the model.

### regressionAccuracy
Calculates the accuracy of a binary regression model.

### Accuracy_Categorical
Calculate the accuracy of model with a categorical cross entropy loss function.

### load_mnist_dataset (function)
function that loads a per-saved model.

### create_data_mnist (function)
function that loads the x and y datasets (inputs and ground truth vectors for training and testing). 

### predict (function)
Uses the model to predict the output of the input. 

## NNTest.png

A custom test image made by me to test the real-world accuracy of the model (the model's output was 'shirt', which is correct). 

## fashion_mnist.model

This is a saved pre-trained neural network that takes in as input 28-by-28 pixel image of clothing (shirts, pants, hats, shoes, etc), and outputs what type of clothing the inputed image is. 
The model has been trained on the fashion MNIST model, a dataset of 28-by-28 pixel images of clothes. This dataset is free to download off the internet. 
