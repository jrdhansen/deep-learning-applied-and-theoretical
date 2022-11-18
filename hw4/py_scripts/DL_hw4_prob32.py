# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:12:47 2019

@author: jrdha
"""














#==============================================================================
#==============================================================================
#==== PROBLEM 3.1 CODE  =======================================================
#==============================================================================
#==============================================================================
#==============================================================================



#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import torch
import torch.nn as nn
import torchvision.datasets as dsets   # contains the MNIST data
import torchvision.transforms as transforms
from torch.autograd import Variable

     





                          
#==============================================================================
#==== CLASSES & FUNCTIONS =====================================================
#==============================================================================

class Net(nn.Module):
    """
    This class initializes and specifies the structure for our neural network.
    The nn.Module is the base class for all neural network modules.
    """
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes, dropout_rate):
        """
        This function initializes the neural network (constructor). We specify
        layer sizes, number of layers, connectedness, and activation
        function(s).
        
        Parameters
        ----------
        input_size : int
            This is the number of neurons in the input layer (784 for us).
        num_classes : int
            This is the number of neurons in the output layer (10 for us).
        
        Returns
        -------
        n/a        
        """
        # Inherited from the parent class nn.Module
        super(Net, self).__init__() 
        # First/only fully-connected layer: 784 (input) --> first hidden layer
        self.fc1 = nn.Linear(input_size, hid_size_1)  
        # Connecting the first hidden layer with the second hidden layer
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)
        # Connecting the second hidden layer with the output layer
        self.fc3 = nn.Linear(hid_size_2, num_classes)
        # Use a ReLU activation function to determine 10-node layer outputs.
        self.Sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        # Inherit drop-out function
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        This function performs the forward passing of the network, stacking the
        layers on top (side-by-side) of each other.
        
        Parameters
        ----------
        x : either vector, matrix, or tensor of floats
            To be honest, I'm not 100% sure what this parameter is. In the code
            below I never explicitly call the "forward" function.
            My best guess is that x is the output of the previous layer (or the
            input in the case of the first layer). This makes since, because we
            take x and then it moves to the output layer after passing through
            the Sigmoid activation function (becomes the first "out").
        
        Returns
        -------
        out : either a vector, matrix, or tensor of floats
            Again, I'm not 100% sure what this is. I'm more confident in saying
            that it is the output of the final layer, but I'm not entirely sure
            whether it's a vector (i.e. the response at each of the 10 nodes 
            for a single image input) or a matrix (i.e. the response at each of
            the 10 nodes for a batch of image inputs).
        """
        # Output of the first set of neurons (just the inputs themselves).
        out = self.dropout(self.fc1(x))
        out = self.Sigmoid(out)
        out = self.dropout(self.fc2(out))
        out = self.Sigmoid(out)
        out = self.fc3(out)
        # The output of the final layer after being passed through the Sigmoid
        # activation function.
        #out = self.Sigmoid(out)
        # Returns the outputs of the final 10-node layer.
        return out
    
    
    
    
    
#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Initialize hyper-parameters
input_size = 784
num_classes = 10
hid_size_1 = 100
hid_size_2 = 100
num_epochs = 5
batch_size = 20
learning_rate = 0.01    
    
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





#==============================================================================
#==== Using CPU only ==========================================================
#==============================================================================

# Instantiate the network (passing the size of the layers).
net_part1 = Net(input_size, hid_size_1, hid_size_2, num_classes,
                dropout_rate=0.1)

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
# The prompt did specify to use stochastic gradient descent, hence the SGD
# optimizer. I left the momentum argument as the default of 0.0.
criterion = nn.CrossEntropyLoss()
# The weight_decay argument is L2 regularzation.
optimizer = torch.optim.SGD(params=net_part1.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=0.0)
#optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

#==== Training the model ======================================================
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
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_dataset)//batch_size, loss.item()))

#==== Testing the model =======================================================
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








































































#==============================================================================
#==============================================================================
#==== PROBLEM 3.2 =============================================================
#==============================================================================
#==============================================================================





#==============================================================================
#==== CLASSES & FUNCTIONS =====================================================
#==============================================================================

# The class that initializes weights with the weights learned in problem 3.1
#---------------------------------------------------------------------------

class Net(nn.Module):
    """
    This class initializes and specifies the structure for our neural network.
    The nn.Module is the base class for all neural network modules.
    """
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes, dropout_rate):
        """
        This function initializes the neural network (constructor). We specify
        layer sizes, number of layers, connectedness, and activation
        function(s).
        
        Parameters
        ----------
        input_size : int
            This is the number of neurons in the input layer (784 for us).
        num_classes : int
            This is the number of neurons in the output layer (10 for us).
        
        Returns
        -------
        n/a        
        """
        # Inherited from the parent class nn.Module
        super(Net, self).__init__() 
        # First/only fully-connected layer: 784 (input) --> first hidden layer
        self.fc1 = nn.Linear(input_size, hid_size_1)  
        # Initialize the weights connecting input layer and hidden_1 randomly
        nn.init.uniform_(self.fc1.weight)
        # Connecting the first hidden layer with the second hidden layer
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)
        # Initialize the weights connecting the hidden_1 and hidden_2 randomly        
        nn.init.uniform_(self.fc2.weight)
        # Connecting the second hidden layer with the output layer
        self.fc3 = nn.Linear(hid_size_2, num_classes)
        # Initialize the weights connecting the hidden_1 and hidden_2 randomly        
        nn.init.uniform_(self.fc2.weight)        
        # Use a ReLU activation function to determine 10-node layer outputs.
        self.Sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        # Inherit drop-out function
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        This function performs the forward passing of the network, stacking the
        layers on top (side-by-side) of each other.
        
        Parameters
        ----------
        x : either vector, matrix, or tensor of floats
            To be honest, I'm not 100% sure what this parameter is. In the code
            below I never explicitly call the "forward" function.
            My best guess is that x is the output of the previous layer (or the
            input in the case of the first layer). This makes since, because we
            take x and then it moves to the output layer after passing through
            the Sigmoid activation function (becomes the first "out").
        
        Returns
        -------
        out : either a vector, matrix, or tensor of floats
            Again, I'm not 100% sure what this is. I'm more confident in saying
            that it is the output of the final layer, but I'm not entirely sure
            whether it's a vector (i.e. the response at each of the 10 nodes 
            for a single image input) or a matrix (i.e. the response at each of
            the 10 nodes for a batch of image inputs).
        """
        # Output of the first set of neurons (just the inputs themselves).
        out = self.dropout(self.fc1(x))
        out = self.Sigmoid(out)
        out = self.dropout(self.fc2(out))
        out = self.Sigmoid(out)
        out = self.fc3(out)
        # The output of the final layer after being passed through the Sigmoid
        # activation function.
        #out = self.Sigmoid(out)
        # Returns the outputs of the final 10-node layer.
        return out



#===== NETWORK THAT INITIALIZES with WEIGHTS OF 3.1 NETWORK ===================
#------------------------------------------------------------------------------


# Instantiate the network, giving it the weights obtained by training the model
# from part 1 of the problem.
# Instantiate the new network WITH THE SAME ARGUMENT VALUES AS FOR net_part1
# NETWORK (they need to have the same architecture).
net_part1_wts = Net(input_size, hid_size_1, hid_size_2, num_classes,
                dropout_rate=0.1)
# This command will give the new network the weights learned in the previous
# network (net_part1 network).
net_part1_wts.load_state_dict(net_part1.state_dict())

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
# The prompt did specify to use stochastic gradient descent, hence the SGD
# optimizer. I left the momentum argument as the default of 0.0.
criterion = nn.CrossEntropyLoss()

# The weight_decay argument is L2 regularzation. FOR THIS NETWORK IT MUST BE
# SET TO 0.0 SO THAT WE'RE NOT DOING BOTH L1 AND L2 REGULARIZATION (WE'RE 
# ALREADY DOING L1 in the for loop below.)
optimizer = torch.optim.SGD(params=net_part1_wts.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=0.0)
#optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

#==== Training the model ======================================================
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Set the gradient back to zero for each mini-batch
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_part1_wts(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        cross_ent_loss = criterion(outputs, labels)
        
        # Define L1 regularization
        # The "all_params" contain all the weights for the respective layers
        # (fc1 = fully connected layer 1, fc2 = fully connected layer 2)
        all_fc1_params = torch.cat([x.view(-1) for x in net_part1_wts.fc1.parameters()])
        all_fc2_params = torch.cat([x.view(-1) for x in net_part1_wts.fc2.parameters()])
        all_fc3_params = torch.cat([x.view(-1) for x in net_part1_wts.fc3.parameters()])
        L1_regrztn = lambda_1 * ((torch.norm(all_fc1_params,1)) +
                                 (torch.norm(all_fc2_params,1)) +
                                 (torch.norm(all_fc3_params,1)))

        # The new loss is cross entropy loss + the regularization term
        loss = cross_ent_loss + L1_regrztn
        
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_dataset)//batch_size, loss.item()))

#==== Testing the model =======================================================
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
    outputs = net_part1_wts(images)
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






















































                          
#==============================================================================
#==== CLASSES & FUNCTIONS =====================================================
#==============================================================================

# The class that initializes weights RANDOMLY 
#--------------------------------------------

class Net(nn.Module):
    """
    This class initializes and specifies the structure for our neural network.
    The nn.Module is the base class for all neural network modules.
    """
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes, dropout_rate):
        """
        This function initializes the neural network (constructor). We specify
        layer sizes, number of layers, connectedness, and activation
        function(s).
        
        Parameters
        ----------
        input_size : int
            This is the number of neurons in the input layer (784 for us).
        num_classes : int
            This is the number of neurons in the output layer (10 for us).
        
        Returns
        -------
        n/a        
        """
        # Inherited from the parent class nn.Module
        super(Net, self).__init__() 
        # First/only fully-connected layer: 784 (input) --> first hidden layer
        self.fc1 = nn.Linear(input_size, hid_size_1)  
        # Initialize the weights connecting input layer and hidden_1 randomly
        nn.init.uniform_(self.fc1.weight)
        # Connecting the first hidden layer with the second hidden layer
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)
        # Initialize the weights connecting the hidden_1 and hidden_2 randomly        
        nn.init.uniform_(self.fc2.weight)
        # Connecting the second hidden layer with the output layer
        self.fc3 = nn.Linear(hid_size_2, num_classes)
        # Initialize the weights connecting the hidden_1 and hidden_2 randomly        
        nn.init.uniform_(self.fc2.weight)        
        # Use a ReLU activation function to determine 10-node layer outputs.
        self.Sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        # Inherit drop-out function
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        This function performs the forward passing of the network, stacking the
        layers on top (side-by-side) of each other.
        
        Parameters
        ----------
        x : either vector, matrix, or tensor of floats
            To be honest, I'm not 100% sure what this parameter is. In the code
            below I never explicitly call the "forward" function.
            My best guess is that x is the output of the previous layer (or the
            input in the case of the first layer). This makes since, because we
            take x and then it moves to the output layer after passing through
            the Sigmoid activation function (becomes the first "out").
        
        Returns
        -------
        out : either a vector, matrix, or tensor of floats
            Again, I'm not 100% sure what this is. I'm more confident in saying
            that it is the output of the final layer, but I'm not entirely sure
            whether it's a vector (i.e. the response at each of the 10 nodes 
            for a single image input) or a matrix (i.e. the response at each of
            the 10 nodes for a batch of image inputs).
        """
        # Output of the first set of neurons (just the inputs themselves).
        out = self.dropout(self.fc1(x))
        out = self.Sigmoid(out)
        out = self.dropout(self.fc2(out))
        out = self.Sigmoid(out)
        out = self.fc3(out)
        # The output of the final layer after being passed through the Sigmoid
        # activation function.
        #out = self.Sigmoid(out)
        # Returns the outputs of the final 10-node layer.
        return out




#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Initialize hyper-parameters
input_size = 784
num_classes = 10
hid_size_1 = 100
hid_size_2 = 100
num_epochs = 5
batch_size = 20
learning_rate = 0.01   
lambda_1 = 0.000001
 
    
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




#===== NETWORK THAT INITIALIZES RANDOMLY ======================================
#------------------------------------------------------------------------------

# Instantiate the network (passing the size of the layers).
net_randomWts = Net(input_size, hid_size_1, hid_size_2,
                    num_classes, dropout_rate=0.1)

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
# The prompt did specify to use stochastic gradient descent, hence the SGD
# optimizer. I left the momentum argument as the default of 0.0.
criterion = nn.CrossEntropyLoss()

# The weight_decay argument is L2 regularzation.
optimizer = torch.optim.SGD(params=net_randomWts.parameters(),
                            lr=learning_rate, momentum=0.9, weight_decay=0.0)
#optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

#==== Training the model ======================================================
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch.Tensor object to Variable: from 784x1 vector to 
        # 28x28 matrix. Convert the lables to Variable as well.
        images = Variable(images.view(-1, 28*28))         
        labels = Variable(labels)
        # Set the gradient back to zero for each mini-batch
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net_randomWts(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.
        cross_ent_loss = criterion(outputs, labels)
        
        # Define L1 regularization
        # The "all_params" contain all the weights for the respective layers
        # (fc1 = fully connected layer 1, fc2 = fully connected layer 2)
        all_fc1_params = torch.cat([x.view(-1) for x in net_randomWts.fc1.parameters()])
        all_fc2_params = torch.cat([x.view(-1) for x in net_randomWts.fc2.parameters()])
        all_fc3_params = torch.cat([x.view(-1) for x in net_randomWts.fc3.parameters()])
        L1_regrztn = lambda_1 * ((torch.norm(all_fc1_params,1)) +
                                 (torch.norm(all_fc2_params,1)) +
                                 (torch.norm(all_fc3_params,1)))

        # The new loss is cross entropy loss + the regularization term
        loss = cross_ent_loss + L1_regrztn
        
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.
        optimizer.step()
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_dataset)//batch_size, loss.item()))

#==== Testing the model =======================================================
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
    outputs = net_randomWts(images)
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





















