import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import cv2
import os 
import pickle
import copy



nnfs.init()

#X is the coordinates of the dataset
#Y is the class of the coordinate (There are 3 classes. the coordinate can either be 1,2 or 3)
 
#object to create neural network layer

# self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
class layer:
    #constructor 
    def __init__(self, numInputs, numNeurons, L1WeightRegularizer=0, L1BaisRegularizer=0, L2WeightRegularizer=0, L2WBaisRegularizer=0):

        self.numInputs = numInputs
        self.numNeurons = numNeurons

        #lambda values to multiply the regularizer values with, to determine by how much we want the regular to affect the learning of the NN
        self.L1WeightRegularizer = L1WeightRegularizer
        self.L1BaisRegularizer = L1BaisRegularizer
        self.L2WeightRegularizer = L2WeightRegularizer
        self.L2WBaisRegularizer = L2WBaisRegularizer

        #creating random weights, the weights will be altered during back progation
        #1st collumn is the weights for the 1st neuron
        self.weights = 0.01*np.random.randn(numInputs, numNeurons)

        #creating the biases for each neuron
        self.baises = np.zeros((1, numNeurons))

    #calculating the activations of the current layer (input of next layer)
    def forward(self, input, training):

        self.input = input
        self.output = np.dot(self.input, self.weights) + self.baises
        #self.prevActivation = input

    def backward(self, currentNodes):
        self.costGradientW = np.dot(self.input.T, currentNodes)
        self.costGradientB = np.sum(currentNodes, axis=0, keepdims=True)

        if self.L1WeightRegularizer > 0:
    
            WeightRegDer = np.ones_like(self.weights)
            WeightRegDer[self.weights < 0] = -1
            self.costGradientW = self.costGradientW + (self.L1WeightRegularizer * WeightRegDer)

        if self.L1BaisRegularizer > 0:
            BaisRegDer = np.ones_like(self.baises)
            BaisRegDer[self.baises < 0] = -1
            lossDerWRTBais = lossDerWRTBais + (self.L1BaisRegularizer * BaisRegDer)

        if self.L2WeightRegularizer > 0:
            #print(self.costGradientW.shape, self.name)
            self.costGradientW = self.costGradientW + (self.L2WeightRegularizer * (2 * self.weights))

        if self.L2WBaisRegularizer > 0:
            self.costGradientB = self.costGradientB + (self.L2WBaisRegularizer * (2 * self.baises))

        self.dinputs = np.dot(currentNodes, self.weights.T)

        #self.lossDerWRTactivation[self.input == 0] = 0

        #print("layer:", type(self.lossDerWRTactivation))
    
    def get_parameters(self):

        return self.weights, self.baises

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.baises = biases


class dropOutReg:

    def __init__(self, rate):
        #getting the amount of neurons to drop
        self.rate = 1 - rate

    def forward(self, inputs, training):
        
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        # getting an array of all the neurons to drop [1,0 ... 1, 0] 1 means keep, zero means drop
        self.percentageofN = np.random.binomial(1, self.rate, size=inputs.shape)/self.rate
        self.output = self.percentageofN * inputs
    
    def backward(self, nodes):
        
        # the partial derivtive of the drop out function is percentageofN, so we are just multiplying the partial derivtives to get the derivtive of the loss wrt the dropout
        self.dinputs = nodes * self.percentageofN
        
#creating ReLuActivation (any value less than 0 becomes 0, above zero values are linear)
class ReLuActivation:
    def forward(self, input, training):
 
        self.input = input
        self.output = np.maximum(0, input)

    def backward(self, dvalues):
        #print(actDer) 
        self.dinputs = dvalues.copy()
        
        self.dinputs[self.input <= 0] = 0

    def predictions(self, outputs):
        return outputs
        
class sigmoidActivation:
    # forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        #getting partial deriviative of the sigmoid function, multiplying by the other partial derivtives to the derivtive of the loss wrt to the sigmoid activation
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class linear_activation:

    def forward(self, inputs, training):
 
        self.inputs = inputs
        self.outputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class softMaxActivation:
    def forward(self, input, training):

        self.inputs = input

        #takes in values, does e^(each value) (so that the value of negative numbers do not get lost)
        #subtract the highest value from each row from each value of that row
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))

        #divide each value from each row with the sum of its row to get the probability distrubution of the firing of neurons 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
        # Flatten output array
            single_output = single_output.reshape(-1, 1)
        # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)

        # Calculate sample-wise gradient
        # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Loss:
    def calculate(self, output, y, *, include_regularization=False):
        #output: the predicted outputs from the NN
        #y: the correct outputs
        sample_losses = self.forward(output, y)

        #calculating the average loss of the network
        data_loss = np.mean(sample_losses)

        #sum of loss per batch
        self.accumulated_sum = self.accumulated_sum + np.sum(sample_losses)
        #the number of losses in each batch
        self.accumulated_count = self.accumulated_count + len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Taking into account over fitting
    # by adding the squared value of the weights, we make sure there is not anny outlying neurons that mostly contribute to the firing of the output
    # by adding this to the loss, we will later add the negative derivative of the regularization, effectivly lowering the loss without overfitting
    def regularization_loss(self):

        regularizationLoss = 0

        for layer in self.trainable_layers:
            # using both L1 and L2 regularization
            if layer.L1WeightRegularizer > 0:
                regularizationLoss = regularizationLoss + (layer.L1WeightRegularizer * np.sum(np.abs(layer.weights)))

            if layer.L1BaisRegularizer > 0:
                regularizationLoss = regularizationLoss + (layer.L1BaisRegularizer * np.sum(np.abs(layer.baises)))

            if layer.L2WeightRegularizer > 0:
                regularizationLoss = regularizationLoss + (layer.L2WeightRegularizer * np.sum(layer.weights * layer.weights))

            if layer.L2WBaisRegularizer > 0:
                regularizationLoss = regularizationLoss + (layer.L2WBaisRegularizer * np.sum(layer.baises * layer.baises))

        return regularizationLoss

    def calculate_accumulated(self, *, include_regularization=False):
        
        #calculating mean loss
        data_loss = self.accumulated_sum/self.accumulated_count

        # if just data loss - return
        if not include_regularization:
            return data_loss
        
        #return data_loss and regularization losses
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):

        #y_pred: outputs from the NN
        # y_true: the wanted outputs  

        # getting the length of the output from the neural network 
        samples = len(y_pred)

        # clipping the outputs at 0.0000001 and 0.9999999
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - (1e-7))

        # if the wanted outputs is a 1-dimensional array
        if len(y_true.shape) == 1:

            # 1D y_true arrays indicate the index of the highest probability in the output for row (sample in the batch)

            #range: iterates through each sample in the batch
            #gets y_pred_clipped[y_true[ith]]

            #print(y_pred_clipped)
            #print(samples)
            #print(y_true)


            correct_condfidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:

            # 2D y_true arrays represents the wanted probability of the batch (the ideal output with perfect confidence) 

            # when you multiply the two matrixes, only the highest probability output will remain since it is being multiplied by one and the other ouputs are being multiplied by 0
            #print(y_true)
            correct_condfidences = np.sum(y_pred_clipped*y_true, axis=1)

        # calculating the negative log of correct confidences
        negative_log_likelihoods = -np.log(correct_condfidences)

        # returning the ngative log of the highest probability output
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We’ll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

            # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):
        
        #clipping the values so that 0/0 does not occur with the predicted values
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # calulating the loss
        sampleLosses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sampleLosses = np.mean(sampleLosses, axis=-1)

        return sampleLosses
    
    def backward(self, dvalues, y_true):

        #getting the partial derivtive wrt to the activation of the output layer 
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples

class momentumSGD:

    # when implemented with momentum, the loss will always go towards the global minimum, if the momentum is not too high. The loss may go past the global minimum, but will
    # come back down, and eventually lose the momentum (not literaly in regards to the NN), and finalize approximately at the global loss

    def __init__(self, learnrate, decay, momentum):

        self.learnrate = learnrate
        self.decay = decay
        self.momentum = momentum

    def applyGradients(self, layer):

        #decaying the learning rate as the iterations go on (reducing the learning rate by some fraction of the learning rate)
        layer.current_learning_rate = self.learnrate * (1. / (1. + self.decay * layer.iterations))

        #getting the gradient of the weights with the momentum of the gradient (this makes sure that the function does not get stuck in a local minimum). The steeper the gradient, the further down the loss goes (similar to ball going down a hill)
        layer.weightUpdates = (self.momentum * layer.weight_momentums) - (layer.current_learning_rate * layer.costGradientW)
        layer.weight_momentums = layer.weightUpdates

        layer.baisUpdates = (self.momentum * layer.baises_momentums) - (layer.current_learning_rate * layer.costGradientB)
        layer.baises_momentums = layer.baisUpdates

        #updating the baises and the weights. We are not subtraction since we are already subtracting when accounting for the momentum
        layer.baises = layer.baises + layer.baisUpdates
        layer.weights = layer.weights + layer.weightUpdates

        #updating the iterations 
        layer.iterations = layer.iterations + 1

class adamOptimizer:

    # I DO NOT UNDERSTAND ADAM OPTIMIZER. IT IS THE MOST EFFECIENT OPTIMIZERR AND MOST USED SO I AM TRUSTING IT. 

    def __init__(self, learnrate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learnrate = learnrate
        self.decay = decay
        self.iterations = 0 
        self.current_learning_rate = learnrate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def beforeGradients(self):
        #decaying the learning rate as the iterations go on (reducing the learning rate by some fraction of the learning rate)
        #print(self.iterations)
        if self.decay:
            self.current_learning_rate = self.learnrate * (1. / (1. + self.decay * self.iterations))


    def applyGradients(self, layer):

        # Update momentum with current gradients

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)

            layer.baises_momentums = np.zeros_like(layer.baises)

            layer.weight_cache = np.zeros_like(layer.weights)

            layer.baises_cache = np.zeros_like(layer.baises)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.costGradientW

        layer.baises_momentums = self.beta_1 * layer.baises_momentums + (1 - self.beta_1) * layer.costGradientB

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        bias_momentums_corrected = layer.baises_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.costGradientW**2

        layer.baises_cache = self.beta_2 * layer.baises_cache + (1 - self.beta_2) * layer.costGradientB**2


        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))

        bias_cache_corrected = layer.baises_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.baises += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        #updating the iterations 
    
    def afterGradient(self):
        self.iterations = self.iterations + 1

#combining softmax activation with categorical cross entropy
class activation_softmax_loss_categoricalCrossentropy():

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples

class loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):

        loss = np.mean((y_true - y_pred)**2, axis=1)

        return loss
    
    def backward(self, dvalues, y_true):

        #number of samples
        samples = len(dvalues)

        #number of outputs 
        outputs = len(dvalues[0])

        #calculating the gradient
        self.dinputs = -2 * (y_true - dvalues) / outputs

        #normalizing the gradient (averaging the gradient across all samples)
        self.dinputs = self.dinputs/samples

class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):

        loss = np.mean(np.abs(y_true - y_pred), axis=-1)

        return loss
    
    def backward(self, dvalues, y_true):
        
        #getting number of samples
        samples = len(dvalues)

        #getting the number of outputs
        outputs = len(dvalues[0])

        #calcutlating the gradient
        self.dinputs = np.sign((y_true - dvalues))/outputs

        #normalixing the hradients (averaging all the gradients )
        self.dinputs = self.dinputs / samples

class Model:
    def __init__(self):

        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):

        #checking if any loss, optimizer, or accuracy object was given
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):

        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1] 
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1] 
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss 
                self.output_layer_activation = self.layers[i]


            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
            
        #if there is no need for optimization
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], softMaxActivation) and (isinstance(self.loss, Loss_CategoricalCrossentropy)):

            self.softmax_classifier_output = activation_softmax_loss_categoricalCrossentropy()

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        
        self.accuracy.init(y)

        #default value if batch size is not set
        train_steps = 1

        #calculate the number of steps that needs to taken per epoch
        if batch_size is not None:
            train_steps = len(X) // batch_size
            #dividing rounds down. If there are some remaining
            #data, but not a full batch, this won't include it
            # Add 1 to include ethis not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1


        for epoch in range(1, epochs+1):

            print(f"epoch: {epoch}")
            
            #reset the accumulated values in loss and accuracy objects
            #preparing to calculate the loss of a new batch
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                #if batch size is not set
                # train using the whole dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                #if batch size is specified, then create a batch
                else:
                    batch_X = X[step*batch_size : (step + 1)*batch_size]
                    batch_y = y[step*batch_size : (step + 1)*batch_size]
            
                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)

                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)

                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.beforeGradients()

                for layer in self.trainable_layers:
                    self.optimizer.applyGradients(layer)
                
                self.optimizer.afterGradient()

                if not step % print_every or step == train_steps - 1:
                    print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:

            self.evaluate(*validation_data, batch_size=batch_size)

        # getting the loss and accuracy per epoch (average loss between all the batches in the epoch)
        epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()

        print(f'training, ' +
              f'acc: {epoch_accuracy:.3f}, ' +
              f'loss: {epoch_loss:.3f} (' +
              f'data_loss: {epoch_data_loss:.3f}, ' +
              f'reg_loss: {epoch_regularization_loss:.3f}), ' +
              f'lr: {self.optimizer.current_learning_rate}')

        #if there is the validation data
       
    def evaluate(self, X_val, y_val, *, batch_size=None):
        
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            #dividing rounds down. If there are some remaining
            #data, but not a full batch, this won't include it
            # Add 1 to include ethis not full batch
            

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()
  
        for step in range(validation_steps):

            #setting the batch size if no batch size
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            #otherwise create a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]


            # Perform the foward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            #get prediction (output of NN)
            predictions = self.output_layer_activation.predictions(output)

            self.accuracy.calculate(predictions, batch_y)

        # get the validation loss and accuracy (average loss and accuracy over the batches)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output
    
    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                #(layer.costGradientW)
                layer.backward(layer.next.dinputs)
            
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def get_parameters(self):

        #create a list for all the parameters for each layer (weights and biases)
        parameters = []

        #iterate over all the trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        #return all the paramters of the layers
        return parameters

    def set_parameters(self, parameters):

        #iterate over all the layers and set the new parameters
        #paramters is an array that contains all the wieghts and biases of each layer
        #each index of paramters is the weights and biases of a layer
        for paramater_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*paramater_set)

    def save_parameters(self, path):

        #open a file in the binary-write mode
        #and save paramters to it
        with open(path, 'wb') as f: #creates a file in binary write mode
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        
        with open(path, 'rb') as f: #opens file in binary read mode
            #setting the layers of the model's parameters
            self.set_parameters(pickle.load(f))

    #first
    def save(self, path):
        
        #make a deep copy of the current model instance
        #(including all the wieghts and baises)
        model = copy.deepcopy(self)

        #resseting the accumulated values in the loss and accuracy objects
        model.loss.new_pass()
        model.loss.new_pass()

        #remove data from the input layer and gradients from loss
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # remove all properties of each layer
        # removing each property listed below
        for layer in model.layers:
            for property in ['input', 'output', 'dinputs', 'costGradientW', 'costGradientB']:
                layer.__dict__.pop(property, None)

        # writing the model to a file in binary-write mode
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    #second
    @staticmethod
    def load(path):

        #open model in binary read mode
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # return a model
        return model

    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_step = 1

        #calculate number of steps
        if batch_size is not None:
            prediction_step = len(X) // batch_size

            if prediction_step * batch_size < len(X):
                prediction_step += 1
        
        # Model Outputs
        outputs = []

        # iterate over steps
        for step in range(prediction_step):

            if batch_size is None:
                batch_X = X

            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            batch_output = self.forward(batch_X, training=False) 

            outputs.append(batch_output)
        
        return np.vstack(outputs)


class Layer_Input:

    def forward(self, inputs, training):
        
        self.output = inputs

class Accuracy:

    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        #add the accumalted sum of matching values and sample count
        self.accumalated_sum += np.sum(comparisons)
        self.accumalated_count += len(comparisons)

        return accuracy
    
    def calculate_accumulated(self):
        
        #calculate the mean accuracy per batch
        accuracy = self.accumalated_sum / self.accumalated_count

        # return accuracy per batch 
        return accuracy

    def new_pass(self):
        self.accumalated_sum = 0
        self.accumalated_count = 0

class regressionAccuracy(Accuracy):
        
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):

        self.binary = binary
    
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

#loads the mnist fashion dataset from the NNLearning folder (this file and the images are in the same directory)
def load_mnist_dataset(dataset, path):

    #loading the labels of the data (all the correct classifications of the images)
    #path is the folder name, dataset is the dataset that needs to be retreived 
    #os.path.join(path, dataset) join path and dataset with a '/'
    labels = os.listdir(os.path.join(path, dataset))

    #input data (all the images)
    X = []
    #correct output (all the labels)
    y = []

    #for each label folder
    for label in labels:

        print(label)

        #for each image in the given folder
        for file in os.listdir(os.path.join(path, dataset, label)):

            #read the image in to the file (in a 2d array of grayscale values)
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            #parallel arrays
            #add image to X
            #add label to y
            X.append(image)
            y.append(label)
    
    #converting the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# getting MNIST dataset (train and test) 
def create_data_mnist(path):

    #load the training and testing data
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # return data
    return X, y, X_test, y_test

def predict(image, model):
    fashion_mnist_labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    #read the image
    image_data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # invert image colors
    image_data = 255 - image_data

    #reshape and scale pixel data (from 0 - 255 to -1 - 1 )
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    #predict the image
    confidences = model.predict(image_data)

    # Get prediction instad of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)

    #index 0 since there is only one image
    predictions = fashion_mnist_labels[predictions[0]]

    return predictions

# create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

#we now need to shuffle the data 
#the model will learn that the quickest way to reduce
#loss is to always predict a 0, since it’ll see several batches of the data with class 0 only. Then,
#between 6,000 and 12,000, the loss will initially spike as the label will change and the model will
#be predicting, incorrectly, still label 0, and will likely learn that now it needs to always predict
#class 1 (as that’s all it sees in batches of labels which we optimize for).
#so without shuffling, the neural network will be baised toward the last class in the dataset

#returns an array from 0-59999
keys = np.array(range(X.shape[0]))
#shuffling keys
np.random.shuffle(keys)

#shuffling both input and correct output in regards to the keys (so that they match)
X = X[keys]
y = y[keys]

# current shape of X (60000, 28, 28)
#the 28, 28 part are the images
#neural networks cannot take 2d arrays as input, so the [28][28] arrays
# must be flattened to [784] array
# resulting dataset (60000, 784)
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# downscaling the 0-255 array images to -1 to 1 array images
# here we knew min = 0, max = 255. In other datasets, the min and max
#can be very different, so we might use a combination of standard deviation and average to 
#find the min and max
X = (X.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


#X2, y2 = spiral_data(samples=100, classes=3)

model = Model.load('fashion_mnist.model')

#this is just an example image. 
print(predict('NNTest.png', model))




