'''
    File name: DL_hw5_prob3.py
    Author: Jared Hansen
    Date created: 03/26/2019
    Date last modified: 03/29/2019
    Python Version: 3.6.4

===============================================================================
CODE FOR HW5, PROLEM 3 IN [DEEP LEARNING: THEORY AND APPLICATIONS]
===============================================================================
'''










'''
-------------------------------------------------------------------------------
---- PROBLEM 3 ----------------------------------------------------------------
-------------------------------------------------------------------------------
PROMPT: 
        For this problem you will reuse the code you wrote for Homework 4 that
        trained a fully connected neural network with 2 hidden layers on the 
        MNIST dataset. You should retain the same number of nodes in each layer
        as in your final dataset. Keep the same values for the tuning
        parameters unless requested otherwise below.
        
        
PART 1: 
        Use the following approaches to optimization:
        (1) standard SGD,
        (2) SGD with momentum
        (3) AdaGrad
        (4) Adam
        - Tune any of the associated parameters including global learning rate
          sing the validation accuracy.
        - Report the final parameters selected in each case, and the final test
          accuracy in each case.
        - Provide two plots with the results from all four approaches:
          (1) the training cost VS the number of epochs
          (2) the validation accuracy VS the number of epochs
        - Which optimization approach seems to be working the best and why?
        
        
PART 2:
        - Pick one of the optimization approaches above.
        - Using the same network, apply batch normalization to each of the 
          hidden layers and re-tune the (global) learning rate using the 
          validation accuracy.
        - Report the new learning rate and the final test accuracy.
        - Does batch normalization seem to help in this case?
'''










#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import torch
import torch.nn as nn
import torchvision.datasets as dsets   # contains the MNIST data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

     
              
#==============================================================================
#==== CLASSES & FUNCTIONS =====================================================
#==============================================================================

class Net(nn.Module):
    """
    This class initializes and specifies the structure for our neural network:
    -- input layer, two hidden layers (30 <= num_nodes <= 100), output layer
    -- implements dropout and L2 regularization
    The nn.Module is the base class for all neural network modules.
    """
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes, drop_i,
                 drop_h):
        """
        This function initializes the neural network (constructor). We specify
        layer sizes, number of layers, connectedness, and activation
        function(s).
        
        Parameters
        ----------
        input_size : int
            This is the number of neurons in the input layer (784 for us).
        hid_size_1 : int
            This is the number of neurons in the first hidden layer.
        hid_size_2 : int
            THis is the number of neurons in the second hidden layer.
        num_classes : int
            This is the number of neurons in the output layer (10 for us).
        drop : float
            This is the proportion of neurons we drop from each layer during
            training. (Must be 0 < drop < 1)
        
        Returns
        -------
        n/a          
        """
        # Inherited from the parent class nn.Module
        super(Net, self).__init__() 
        # First fully-connected layer: 784 (input) --> first hidden layer
        self.fc1 = nn.Linear(input_size, hid_size_1)  
        # Connecting the first hidden layer with the second hidden layer
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)
        # Connecting the second hidden layer with the output layer
        self.fc3 = nn.Linear(hid_size_2, num_classes)
        # Inherit activation functions for designing network.
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        # The dropout_i method is meant to be used with the input layer
        # (hence the "i") with neurons being dropped with probability drop_i.
        self.dropout_i = nn.Dropout(drop_i)
        # The dropout_h method is meant to be used with the hidden layers
        # (hence the "h") with neurons being dropped with probability drop_h.        
        self.dropout_h = nn.Dropout(drop_h)
    
    def forward(self, x):
        """
        This function performs the forward passing of the network, stacking the
        layers on top (side-by-side) of each other, computing values and
        passing those values to the next layer.
        
        Parameters
        ----------
        x : either vector, matrix, or tensor of floats
            To be honest, I'm not 100% sure what this parameter is. In the code
            below I never explicitly call the "forward" function.
            My best guess is that x is the input of the case of the first
            layer. This makes sense, because we take x and then it moves to the
            hidden layers and then output layer after passing through
            the Sigmoid activation function (becomes the successive "out").
        
        Returns
        -------
        out : either a vector, matrix, or tensor of floats
            Again, I'm not 100% sure what this is. I'm more confident in saying
            that it is the output of the final layer, but I'm not entirely sure
            whether it's a vector (i.e. the response at each of the 10 nodes 
            for a single image input) or a matrix (i.e. the response at each of
            the 10 nodes for a batch of image inputs).
            If I had to guess, I'd say that this is a tensor.
        """
        # Connect the input neurons with the first hidden layer.
        # Here we implement dropout of neurons in the first hidden layer.
        out = self.dropout_i(self.fc1(x))
        # Apply sigmoid activation function to z's (weighted sums) of first
        # hidden layer.
        out = self.Sigmoid(out)
        # Connect the hidden_1 layer with the hidden_2 layer.
        # Here we implement dropout of neurons in the second hidden layer.
        out = self.dropout_h(self.fc2(out))
        # Apply sigmoid activation function to z's (weighted sums) of second
        # hidden layer.        
        out = self.Sigmoid(out)
        # Connect the hidden_2 layer with the output layer.
        out = self.fc3(out)
        #out = self.Sigmoid(out)
        #out = self.softmax(out)
        #out = self.relu(out)
        
        # Returns the outputs of the final 10-node layer.
        return out
    
    
    
   
    
    
    
    
    
    
    
    
'''
-------------------------------------------------------------------------------
---- PROBLEM 3.1 --------------------------------------------------------------
-------------------------------------------------------------------------------
Use the following approaches to optimization:
(1) standard SGD,
(2) SGD with momentum
(3) AdaGrad
(4) Adam
- Tune any of the associated parameters including global learning rate
  sing the validation accuracy.
- Report the final parameters selected in each case, and the final test
  accuracy in each case.
- Provide two plots with the results from all four approaches:
  (1) the training cost VS the number of epochs
  (2) the validation accuracy VS the number of epochs
- Which optimization approach seems to be working the best and why?
'''    
    

    
    
#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Initialize hyper-parameters (fixed)
input_size = 784
num_classes = 10
drop_i = 0.0
drop_h = 0.0
num_epochs = 50

# Initialize hyper-parameters (tuned)
#learning_rate = 5.0
learning_rate = 0.05
L2_param = 0.0
batch_size = 20
hid_size_1 = 100
hid_size_2 = 100
    
# Download the MNIST data (using library imported above)
train_dataset = dsets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Load the MNIST data (using library imported above)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#===== Create [validation set + new training set] =============================
# Per the prompt instructions we need to create a validation data set.
# For this, I'm just going to take the first 10,000 samples as validation,
# leaving the remaining 50,000 as the training data. After iteratively tuning
# using the training and validation sets, I'll train on the full train_dataset.
# Since the train_loader shuffled the data, we don't need to re-shuffle again.
train_small, validation = torch.utils.data.random_split(train_dataset, [50000, 10000])
# Load in the new (smaller) training data
new_train_loader = torch.utils.data.DataLoader(dataset=train_small,
                                               batch_size=batch_size,
                                               shuffle=False)
# Load in the validation data
val_loader = torch.utils.data.DataLoader(dataset=validation,
                                         batch_size=batch_size,
                                         shuffle=False)



#==============================================================================
#==== USING CPU ONLY ==========================================================
#==============================================================================

# Instantiate the network (passing the size of the layers and dropout rate).
net_part1 = Net(input_size, hid_size_1, hid_size_2, num_classes, drop_i, drop_h)

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
criterion = nn.CrossEntropyLoss()

#------------------------------------------------------------------------------
=# UNCOMMENT WHICHEVER OPTIMIZER YOU WANT TO USE FOR TRAIN/VALIDATION/TEST.
#------------------------------------------------------------------------------
#optimizer = torch.optim.SGD(params=net_part1.parameters(), lr=0.8, momentum=0.0)
#optimizer = torch.optim.SGD(params=net_part1.parameters(), lr=0.05, momentum=0.9)
#optimizer = torch.optim.Adagrad(params=net_part1.parameters(), lr=0.1, lr_decay=1e-8)
optimizer = torch.optim.Adam(params=net_part1.parameters(), lr=0.003, betas=[0.9, 0.999], eps=1e-8, amsgrad=False)



#==== TRAINING THE MODEL, PT I ================================================
#==== This chunk of training + validation is used for parameter tuning. 
#==============================================================================
# Initialize an empty list to store the training costs at the end of each epoch
train_costs = []
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(new_train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_part1(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        loss = criterion(outputs, labels)
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This checks to see if we're at the end of an epoch. If so, it stores
        # the current value of the cost function in the proper list.
        if(i+1 == len(train_small)//batch_size):
            train_costs.append(loss.item())
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_small)//batch_size, loss.item()))
# Plot the training cost VS number of epochs
plt.plot(train_costs)            
# Write out the values of [the cost at the end of each epoch] to a file
file_path = "C:/__JARED/_USU_Sp2019/_Moon_DeepLearning/hw5/"
with open(file_path + 'trainCosts_SGD_plain.txt', 'w') as f:
    for item in train_costs:
        f.write("%s\n" % item) 
        
            
#==== VALIDATING (TUNING) THE MODEL ===========================================
#==== Used in conjunction with the TRAINING PT I above for parameter tuning.
#==============================================================================
# Initialize the number of correct predictions and the total num of test images
correct = 0
total = 0
# We'll use the trained network to predict the value for each test image (so 
# we iterate over all test image/label pairs in the test data).
for images, labels in val_loader:
    # Convert torch.Tensor object to Variable: from 784x1 vector to 
    # 28x28 matrix.
    images = Variable(images.view(-1, 28*28))
    # Determine predicted output (0,1,2,...,9) for each image in the test data
    # by passing them through the trained network.
    outputs = net_part1(images)
    # Determine the predicted label for this image by choosing the node with
    # the highest (max) activation in the output layer.
    _, predicted = torch.max(outputs.data, 1)
    # Increment the total variable by 1 each time through the loop (total is a
    # simple int object).
    total += labels.size(0)
    # Increment correct by 1 each time through the loop if "predicted" is the
    # same as the true "labels" for that image. (This is a torch.Tensor.)
    correct += (predicted == labels).sum()
    
# Print the validation accuracy of the network with the current parameters.
print("Validation accuracy: ", round(100*(correct.item()/total), 3), "%", sep="")




#==== TRAINING THE MODEL, PT. II ==============================================
# After having used validation to tune hyperparameters, go back and train on
# the original training data (to have 10,000 additional data points).
#==============================================================================
# Initialize an empty list to store the training costs at the end of each epoch
train_costs = []
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_part1(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        loss = criterion(outputs, labels)
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This checks to see if we're at the end of an epoch. If so, it stores
        # the current value of the cost function in the proper list.
        if(i+1 == len(train_small)//batch_size):
            train_costs.append(loss.item())
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        '''
        I'VE COMMENTED THIS OUT TO SAVE TIME ON TRAINING FOR LARGER NUMBER 
        OF EPOCHS
        ----------------------------------------------------------------------
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_dataset)//batch_size, loss.item()))
        '''
# Plot the training cost VS number of epochs
plt.plot(train_costs)            
# Write out the costs to a file
file_path = "C:/__JARED/_USU_Sp2019/_Moon_DeepLearning/hw5/"
with open(file_path + 'trainCosts_Adam.txt', 'w') as f:
    for item in train_costs:
        f.write("%s\n" % item) 


#==== TESTING THE MODEL =======================================================
#==============================================================================
# Initialize the number of correct predictions and the total num of test images
correct = 0
total = 0
# We'll use the trained network to predict the value for each test image (so 
# we iterate over all test image/label pairs in the test data).
for images, labels in test_loader:
    # Convert torch.Tensor object to Variable: from 784x1 vector to 
    # 28x28 matrix.
    images = Variable(images.view(-1, 28*28))
    # Determine predicted output (0,1,2,...,9) for each image in the test data
    # by passing them through the trained network.
    outputs = net_part1(images)
    # Determine the predicted label for this image by choosing the node with
    # the highest (max) activation in the output layer.
    _, predicted = torch.max(outputs.data, 1)
    # Increment the total variable by 1 each time through the loop (total is a
    # simple int object).
    total += labels.size(0)
    # Increment correct by 1 each time through the loop if "predicted" is the
    # same as the true "labels" for that image. (This is a torch.Tensor.)
    correct += (predicted == labels).sum()
    
# Print the test accuracy of the network.
print("Test accuracy: ", round(100*(correct.item()/total), 3), "%", sep="")








'''
Code for generating the data of "the validation accuracy VS num epochs"
'''
#==== TRAINING + VALIDATING  TOGETHER =========================================
#==============================================================================
# Initialize an empty list to store the validation accuracy at the end of
# each epoch
val_accs = []
# We'll iterate over the data num_epochs times while training
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(new_train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_part1(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        loss = criterion(outputs, labels)
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This checks to see if we're at the end of an epoch. If so, it 
        # calculates the validation accuracy for the current version of the 
        # model and stores it in the proper list (this happens num_epocsh times).
        if(i+1 == len(train_small)//batch_size):
            # Initialize the number of correct predictions and the total num of test images
            correct = 0
            total = 0
            # We'll use the trained network to predict the value for each test image (so 
            # we iterate over all test image/label pairs in the test data).
            for images, labels in val_loader:
                # Convert torch.Tensor object to Variable: from 784x1 vector to 
                # 28x28 matrix.
                images = Variable(images.view(-1, 28*28))
                # Determine predicted output (0,1,2,...,9) for each image in the test data
                # by passing them through the trained network.
                outputs = net_part1(images)
                # Determine the predicted label for this image by choosing the node with
                # the highest (max) activation in the output layer.
                _, predicted = torch.max(outputs.data, 1)
                # Increment the total variable by 1 each time through the loop (total is a
                # simple int object).
                total += labels.size(0)
                # Increment correct by 1 each time through the loop if "predicted" is the
                # same as the true "labels" for that image. (This is a torch.Tensor.)
                correct += (predicted == labels).sum()
                
            # Print the validation accuracy of the network with the current parameters.
            valid_accuracy = round(100*(correct.item()/total), 3)
            print("Validation accuracy: ", valid_accuracy, "%", sep="")
            val_accs.append(valid_accuracy)
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        '''
        I'VE COMMENTED THIS OUT TO SAVE TIME ON TRAINING FOR LARGER NUMBER 
        OF EPOCHS
        ----------------------------------------------------------------------
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_dataset)//batch_size, loss.item()))
        '''
# Plot the training cost VS number of epochs
plt.plot(val_accs)            
# Write out the validation accuracy at the end of each epoch to a file
file_path = "C:/__JARED/_USU_Sp2019/_Moon_DeepLearning/hw5/"
with open(file_path + 'val_ACCs_Adam.txt', 'w') as f:
    for item in val_accs:
        f.write("%s\n" % item) 



#==============================================================================
#==== CREATING THE TWO DESIRED PLOTS ==========================================
#==============================================================================
        
# In order to not have to have the 8 arrays needed for plotting stored in 
# memory (i.e. needing to have done hours-worth of training all in one session)
# I need to read back in all 8 of the arrays that I wrote out. Then we can plot
# them.

# Read in all 8 of the needed arrays (train cost VS epochs, valAcc VS epochs)
trainCosts_SGD_plain = np.loadtxt(file_path + "trainCosts_SGD_plain.txt")
trainCosts_SGD_momentum = np.loadtxt(file_path + "trainCosts_SGD_momentum.txt")
trainCosts_AdaGrad = np.loadtxt(file_path + "trainCosts_AdaGrad.txt")
trainCosts_Adam = np.loadtxt(file_path + "trainCosts_Adam.txt")
valACCs_SGD_plain = np.loadtxt(file_path + "val_ACCs_SGD_plain.txt")
valACCs_SGD_momentum = np.loadtxt(file_path + "val_ACCs_SGD_momentum.txt")
valACCs_AdaGrad = np.loadtxt(file_path + "val_ACCs_AdaGrad.txt")
valACCs_Adam = np.loadtxt(file_path + "val_ACCs_Adam.txt")

# Create the plot of [TRAINING COST vs NUMBER OF EPOCHS]
plt.plot(trainCosts_SGD_plain, color = "blue", linewidth = 2,
         label = "trainCosts_SGD_plain")
plt.plot(trainCosts_SGD_momentum, color = "red", linewidth = 2,
         label = "trainCosts_SGD_momentum")
plt.plot(trainCosts_AdaGrad, color = "black", linewidth = 2,
         label = "trainCosts_AdaGrad")
plt.plot(trainCosts_Adam, color = "lime", linewidth = 2,
         label = "TtrainCosts_Adam")
plt.legend()
plt.ylabel("Value of Cost Function during Training")
plt.xlabel("Number of Epochs")
plt.title("Values of Cost Function (while training) VS Number of Epochs")
plt.show()

# Create the plot of [VALIDATION ACCURACY vs NUMBER OF EPOCHS]
plt.plot(valACCs_SGD_plain, color = "blue", linewidth = 2,
         label = "valACCs_SGD_plain")
plt.plot(valACCs_SGD_momentum, color = "red", linewidth = 2,
         label = "valACCs_SGD_momentum")
plt.plot(valACCs_AdaGrad, color = "black", linewidth = 2,
         label = "valACCs_AdaGrad")
plt.plot(valACCs_Adam, color = "lime", linewidth = 2,
         label = "valACCs_Adam")
plt.legend()
plt.ylabel("Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.title("Validation Accuracy VS Number of Epochs")
plt.show()





























'''
-------------------------------------------------------------------------------
---- PROBLEM 3.2 --------------------------------------------------------------
-------------------------------------------------------------------------------
- Pick one of the optimization approaches above.
- Using the same network, apply batch normalization to each of the 
  hidden layers and re-tune the (global) learning rate using the 
  validation accuracy.
- Report the new learning rate and the final test accuracy.
- Does batch normalization seem to help in this case?
'''  




# We need to create a new class of networks that PERFORM BATCH NORMALIZATION
#------------------------------------------------------------------------------
class Net_batchNorm(nn.Module):
    """
    This class initializes and specifies the structure for our neural network:
    -- input layer, two hidden layers (30 <= num_nodes <= 100), output layer
    -- implements dropout and L2 regularization
    The nn.Module is the base class for all neural network modules.
    AND THIS IS DIFFERENT THAN ABOVE: does batch normalization
    """
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes, drop_i,
                 drop_h):
        """
        This function initializes the neural network (constructor). We specify
        layer sizes, number of layers, connectedness, and activation
        function(s).
        
        Parameters
        ----------
        input_size : int
            This is the number of neurons in the input layer (784 for us).
        hid_size_1 : int
            This is the number of neurons in the first hidden layer.
        hid_size_2 : int
            THis is the number of neurons in the second hidden layer.
        num_classes : int
            This is the number of neurons in the output layer (10 for us).
        drop : float
            This is the proportion of neurons we drop from each layer during
            training. (Must be 0 < drop < 1)
        
        Returns
        -------
        n/a          
        """
        # Inherited from the parent class nn.Module
        super(Net_batchNorm, self).__init__() 
        # First fully-connected layer: 784 (input) --> first hidden layer
        self.fc1 = nn.Linear(input_size, hid_size_1)  
        # Normalize the inputs for the first hidden layer
        self.bn1 = nn.BatchNorm1d(num_features=hid_size_1)
        # Connecting the first hidden layer with the second hidden layer
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)
        # Normalize the inputs for the second layer
        self.bn2 = nn.BatchNorm1d(num_features=hid_size_2)
        # Connecting the second hidden layer with the output layer
        self.fc3 = nn.Linear(hid_size_2, num_classes)
        # Inherit activation functions for designing network.
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        # The dropout_i method is meant to be used with the input layer
        # (hence the "i") with neurons being dropped with probability drop_i.
        self.dropout_i = nn.Dropout(drop_i)
        # The dropout_h method is meant to be used with the hidden layers
        # (hence the "h") with neurons being dropped with probability drop_h.        
        self.dropout_h = nn.Dropout(drop_h)
    
    def forward(self, x):
        """
        This function performs the forward passing of the network, stacking the
        layers on top (side-by-side) of each other, computing values and
        passing those values to the next layer.
        
        Parameters
        ----------
        x : either vector, matrix, or tensor of floats
            To be honest, I'm not 100% sure what this parameter is. In the code
            below I never explicitly call the "forward" function.
            My best guess is that x is the input of the case of the first
            layer. This makes sense, because we take x and then it moves to the
            hidden layers and then output layer after passing through
            the Sigmoid activation function (becomes the successive "out").
        
        Returns
        -------
        out : either a vector, matrix, or tensor of floats
            Again, I'm not 100% sure what this is. I'm more confident in saying
            that it is the output of the final layer, but I'm not entirely sure
            whether it's a vector (i.e. the response at each of the 10 nodes 
            for a single image input) or a matrix (i.e. the response at each of
            the 10 nodes for a batch of image inputs).
            If I had to guess, I'd say that this is a tensor.
        """
        # Connect the input neurons with the first hidden layer.
        # Here we implement dropout of neurons in the first hidden layer.
        out = self.dropout_i(self.fc1(x))
        # Apply batch normalization to the first hidden layer
        out = self.bn1(out)
        # Apply sigmoid activation function to z's (weighted sums) of first
        # hidden layer.
        out = self.Sigmoid(out)
        # Connect the hidden_1 layer with the hidden_2 layer.
        # Here we implement dropout of neurons in the second hidden layer.
        out = self.dropout_h(self.fc2(out))
        # Apply batch normalization to the second hidden layer
        out = self.bn2(out)
        # Apply sigmoid activation function to z's (weighted sums) of second
        # hidden layer.        
        out = self.Sigmoid(out)
        # Connect the hidden_2 layer with the output layer.
        out = self.fc3(out)
        #out = self.Sigmoid(out)
        #out = self.softmax(out)
        #out = self.relu(out)
        
        # Returns the outputs of the final 10-node layer.
        return out
    
    


#==============================================================================
#==== USING CPU ONLY ==========================================================
#==============================================================================

# Instantiate the network (passing the size of the layers and dropout rate).
# THIS IS THE NEW TYPE OF NET WE DEFINED THAT WILL DO BATCH NORMALIZATION.
net_part2 = Net_batchNorm(input_size, hid_size_1, hid_size_2, num_classes, drop_i, drop_h)

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
criterion = nn.CrossEntropyLoss()

#------------------------------------------------------------------------------
# UNCOMMENT WHICHEVER OPTIMIZER YOU WANT TO USE FOR TRAIN/VALIDATION/TEST.
#------------------------------------------------------------------------------
optimizer = torch.optim.SGD(params=net_part2.parameters(), lr=2.0, momentum=0.0)
#optimizer = torch.optim.SGD(params=net_part2.parameters(), lr=0.05, momentum=0.9)
#optimizer = torch.optim.Adagrad(params=net_part2.parameters(), lr=0.1, lr_decay=1e-8)
#optimizer = torch.optim.Adam(params=net_part2.parameters(), lr=0.003, betas=[0.9, 0.999], eps=1e-8, amsgrad=False)

# Initialize hyper-parameters (fixed)
num_epochs = 50

#==== TRAINING THE MODEL, PT I ================================================
#==== This chunk of training + validation is used for parameter tuning. 
#==============================================================================
# Initialize an empty list to store the training costs at the end of each epoch
train_costs = []
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(new_train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_part2(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        loss = criterion(outputs, labels)
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This checks to see if we're at the end of an epoch. If so, it stores
        # the current value of the cost function in the proper list.
        if(i+1 == len(train_small)//batch_size):
            train_costs.append(loss.item())
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_small)//batch_size, loss.item()))
# Plot the training cost VS number of epochs
plt.plot(train_costs)  


#==== VALIDATING (TUNING) THE MODEL ===========================================
#==== Used in conjunction with the TRAINING PT I above for parameter tuning.
#==============================================================================
# Initialize the number of correct predictions and the total num of test images
correct = 0
total = 0
# We'll use the trained network to predict the value for each test image (so 
# we iterate over all test image/label pairs in the test data).
for images, labels in val_loader:
    # Convert torch.Tensor object to Variable: from 784x1 vector to 
    # 28x28 matrix.
    images = Variable(images.view(-1, 28*28))
    # Determine predicted output (0,1,2,...,9) for each image in the test data
    # by passing them through the trained network.
    outputs = net_part2(images)
    # Determine the predicted label for this image by choosing the node with
    # the highest (max) activation in the output layer.
    _, predicted = torch.max(outputs.data, 1)
    # Increment the total variable by 1 each time through the loop (total is a
    # simple int object).
    total += labels.size(0)
    # Increment correct by 1 each time through the loop if "predicted" is the
    # same as the true "labels" for that image. (This is a torch.Tensor.)
    correct += (predicted == labels).sum()
    
# Print the validation accuracy of the network with the current parameters.
print("Validation accuracy: ", round(100*(correct.item()/total), 3), "%", sep="")



#==== TRAINING THE MODEL, PT. II ==============================================
# After having used validation to tune hyperparameters, go back and train on
# the original training data (to have 10,000 additional data points).
#==============================================================================
# Initialize an empty list to store the training costs at the end of each epoch
train_costs = []
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_part2(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        loss = criterion(outputs, labels)
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This checks to see if we're at the end of an epoch. If so, it stores
        # the current value of the cost function in the proper list.
        if(i+1 == len(train_small)//batch_size):
            train_costs.append(loss.item())
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        '''
        I'VE COMMENTED THIS OUT TO SAVE TIME ON TRAINING FOR LARGER NUMBER 
        OF EPOCHS
        ----------------------------------------------------------------------
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_dataset)//batch_size, loss.item()))
        '''
# Plot the training cost VS number of epochs
plt.plot(train_costs)            


#==== TESTING THE MODEL =======================================================
#==============================================================================
# Initialize the number of correct predictions and the total num of test images
correct = 0
total = 0
# We'll use the trained network to predict the value for each test image (so 
# we iterate over all test image/label pairs in the test data).
for images, labels in test_loader:
    # Convert torch.Tensor object to Variable: from 784x1 vector to 
    # 28x28 matrix.
    images = Variable(images.view(-1, 28*28))
    # Determine predicted output (0,1,2,...,9) for each image in the test data
    # by passing them through the trained network.
    outputs = net_part2(images)
    # Determine the predicted label for this image by choosing the node with
    # the highest (max) activation in the output layer.
    _, predicted = torch.max(outputs.data, 1)
    # Increment the total variable by 1 each time through the loop (total is a
    # simple int object).
    total += labels.size(0)
    # Increment correct by 1 each time through the loop if "predicted" is the
    # same as the true "labels" for that image. (This is a torch.Tensor.)
    correct += (predicted == labels).sum()
    
# Print the test accuracy of the network.
print("Final TEST accuracy using BATCH NORMALIZATION: ",
      round(100*(correct.item()/total), 3), "%", sep="")



