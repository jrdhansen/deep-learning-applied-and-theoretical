'''
    File name: hw4_prob3.py
    Author: Jared Hansen
    Date created: 02/26/2019
    Date last modified: 02/28/2019
    Python Version: 3.6.4

===============================================================================
CODE FOR HW4, PROLEM 3 IN [DEEP LEARNING: THEORY AND APPLICATIONS]
===============================================================================
'''










'''
-------------------------------------------------------------------------------
---- PROBLEM 3.1 --------------------------------------------------------------
-------------------------------------------------------------------------------
PROMPT: Using either Tensorflow or Pytorch, design a single neural network for
        classifying the MNIST dataset.
        -- The neural network must have 2 hidden layers. In other words, your
           final network will have 4 layers total: the input layer, 2 hidden
           layers, and then the output layer.
        -- You will need to select the exact number of nodes in the hidden
           layers by tuning. However, you shouldn't chose less than 30 nodes or 
           more than 100 in each of the hidden layers.
        -- Use dropout and L2 regularization on the weights when training the
           network. Describe your entire design procedure. In particular, make 
           sure to report the following:
        (a) The dropout rate (i.e. the percentage of noeds dropped out each
            time), the activation functions in each layer, cost functions,
            weight initialization strategy, and stopping criterion. These do
            not need to be tuned but you should provide some justification for
            each choice.
        (b) Learning rate, regularization parameter, mini-batch size, and the
            number of nodes in each of the hidden layers. Each of these should
            be tuned in some fashion using a validation data set. Describe your
            tuning procedure for these parameters.
        (c) The final test error. To get full credit for this problem, you will
            need to obtain a test accuracy greater than 98% as this was the
            accuracy obtained using a single hidden leayer with regularization.
            Partial credit will be awarded for designs that do not perform as
            well.
        (d) Include all of your code.
'''




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
batch_size = 16
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
# The weight_decay argument implements L2 regularzation.
optimizer = torch.optim.SGD(params=net_part1.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=L2_param)
#optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)



#==== TRAINING THE MODEL, PT I ================================================
#==============================================================================
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
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1,
                    len(train_small)//batch_size, loss.item()))
            
            
            
#==== VALIDATING (TUNING) THE MODEL ===========================================
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
#==============================================================================
# After having used validation to tune hyperparameters, go back and train on
# the original training data (to have 10,000 additional data points).
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
-------------------------------------------------------------------------------
---- PROBLEM 3.2 --------------------------------------------------------------
-------------------------------------------------------------------------------
PROMPT: Starting with the network you trained in the previous problem, replace
        L2 regularization with L1 regularization and tune the regularization
        parameter as well as the learning rate.
        Use two initialization strategies:
        (1) Initialize the weights obtained using L2 regularization 
        (2) Initialize randomly
        Which initialization strategy worked the best? Based on your results,
        which regularization worked best on this data?
'''




#==============================================================================
#===== NETWORK THAT INITIALIZES with LEARNED WEIGHTS OF PART 3.1 NETWORK ======
#==============================================================================


# Set the regularization parameter (for L1 regularization) and learning rate.
# Will use smaller numbers of epochs for initial tuning, then bump it up after
# achieving good results.
lambda_1 = 0.0
learning_rate = 0.05
num_epochs = 7


# Instantiate the new network WITH THE SAME ARGUMENT VALUES AS FOR net_part1
# NETWORK (they need to have the same architecture).
net_part1_wts = Net(input_size, hid_size_1, hid_size_2, num_classes, drop_i, drop_h)
# This command will give the new network (net_part1_wts) the weights learned in
# the previous network (net_part1 network) in problem 3.1.
net_part1_wts.load_state_dict(net_part1.state_dict())

# I'm using the same loss function as for the previous network.
criterion = nn.CrossEntropyLoss()

# The weight_decay argument is L2 regularzation. FOR THIS NETWORK IT MUST BE
# SET TO 0.0 SO THAT WE'RE NOT DOING BOTH L1 AND L2 REGULARIZATION (WE'RE 
# ALREADY DOING L1 in the for loop below.)
optimizer = torch.optim.SGD(params=net_part1_wts.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=0.0)

#==== TRAINING THE MODEL, PART I ==============================================
# Initially we'll train on the 50,000 observations set aside for training. Then
# we'll validate this trained model on 10,000 obs for validation. After
# obtaining good hyperparameters we'll go back and fit on all 60,000 training
# observations.
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(new_train_loader):
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
        
        #== DEFINE L1 REGULARIZATION ==========================================
        # The "all_fc_params" contain all the weights for the respective layers
        # (i.e. all_fc1_params contains all weights connecting input w/hidden1)
        all_fc1_params = torch.cat([x.view(-1) for x in net_part1_wts.fc1.parameters()])
        all_fc2_params = torch.cat([x.view(-1) for x in net_part1_wts.fc2.parameters()])
        all_fc3_params = torch.cat([x.view(-1) for x in net_part1_wts.fc3.parameters()])
        # The regularization term is the parameter lambda_1 * the norm of
        # the weights (will be inducing sparsity in the weights).
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


#==== VALIDATING (TUNING) THE MODEL ===========================================
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
    
# Print the validation accuracy of the network for the current parameters for
# learning rate and lambda_1 (L1 regularization parameter).
print("Validation accuracy: ", round(100*(correct.item()/total), 3), "%", sep="")


#==== TRAINING THE MODEL, PART II =============================================
# After having used validation to tune hyperparameters, go back and train on
# the original training data (to have 10,000 additional data points).
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
        
        #== DEFINE L1 REGULARIZATION ==========================================
        # The "all_fc_params" contain all the weights for the respective layers
        # (i.e. all_fc1_params contains all weights connecting input w/hidden1)
        all_fc1_params = torch.cat([x.view(-1) for x in net_part1_wts.fc1.parameters()])
        all_fc2_params = torch.cat([x.view(-1) for x in net_part1_wts.fc2.parameters()])
        all_fc3_params = torch.cat([x.view(-1) for x in net_part1_wts.fc3.parameters()])
        # The regularization term is the parameter lambda_1 * the norm of
        # the weights (will be inducing sparsity in the weights).
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


#==== TESTING THE MODEL =======================================================
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
#===== NETWORK THAT INITIALIZES RANDOMLY ======================================
#==============================================================================
                          
#==============================================================================
#==== CLASSES & FUNCTIONS: initializing weights randomly ======================
#==============================================================================

class Net_randomWts(nn.Module):
    """
    This class initializes weights randomly and specifies the structure for our
    neural network.
    The nn.Module is the base class for all neural network modules.
    """
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes, drop):
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
        super(Net_randomWts, self).__init__() 
        # First fully-connected layer: 784 (input) --> first hidden layer
        self.fc1 = nn.Linear(input_size, hid_size_1)  
        # Initialize the weights connecting input layer and hidden_1 randomly
        nn.init.uniform_(self.fc1.weight)
        # Connecting the first hidden layer with the second hidden layer
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)
        # Initialize the weights connecting the hidden_1 and hidden_2 randomly        
        nn.init.uniform_(self.fc2.weight)
        # Connecting the second hidden layer with the output layer
        self.fc3 = nn.Linear(hid_size_2, num_classes)
        # Initialize the weights connecting the hidden_2 and output randomly        
        nn.init.uniform_(self.fc3.weight)        
        # Inherit sigmoid activation function for network design.
        self.Sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        # Inherit drop-out function
        self.dropout = nn.Dropout()
    
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
            output layer after passing through the hidden layers and activation
            functions. (Pass the successive "out" values to next layer.)
        
        Returns
        -------
        out : either a vector, matrix, or tensor of floats
              Again, I'm not 100% sure what this is. I'm more confident in
              saying that it is the output of the final layer, but I'm not
              entirely sure whether it's a vector (i.e. the response at each of
              the 10 nodes for a single image input) or a tensor (i.e. the
              response at each of the 10 nodes for a batch of image inputs).
              If I had to guess, I'd say that this is a tensor.
        """
        # Feeding forward network values (see more detailed comments in the
        # code for problem 1, definition of Net class. Very similar structure.)
        out = self.dropout(self.fc1(x))
        out = self.Sigmoid(out)
        out = self.dropout(self.fc2(out))
        out = self.Sigmoid(out)
        out = self.fc3(out)
        return out




#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# No-tuning-necessary hyper-parameters
input_size    = 784
num_classes   = 10
hid_size_1    = 100
hid_size_2    = 100
batch_size    = 16

# Tuning-necessary hyper-parameters
learning_rate = 0.5
lambda_1      = 0.0
drop_rate     = 0.0
num_epochs    = 7



# Instantiate the network (passing the size of the layers).
net_randomWts = Net_randomWts(input_size, hid_size_1, hid_size_2, num_classes, drop_rate)

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
# The prompt did specify to use stochastic gradient descent, hence the SGD
# optimizer. I left the momentum argument as the default of 0.0.
criterion = nn.CrossEntropyLoss()

# The weight_decay argument is L2 regularzation. FOR THIS NETWORK IT MUST BE
# SET TO 0.0 SO THAT WE'RE NOT DOING BOTH L1 AND L2 REGULARIZATION (WE'RE 
# ALREADY DOING L1 in the for loop below.)
optimizer = torch.optim.SGD(params=net_randomWts.parameters(),
                            lr=learning_rate, momentum=0.9, weight_decay=0.0)
#optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

#==== TRAINING THE MODEL, PART I ==============================================
# Initially we'll train on the 50,000 observations set aside for training. Then
# we'll validate this trained model on 10,000 obs for validation. After
# obtaining good hyperparameters we'll go back and fit on all 60,000 training
# observations.
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(new_train_loader):
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
        
        #== DEFINE L1 REGULARIZATION ==========================================
        # The "all_fc_params" contain all the weights for the respective layers
        # (i.e. all_fc1_params contains all weights connecting input w/hidden1)
        all_fc1_params = torch.cat([x.view(-1) for x in net_randomWts.fc1.parameters()])
        all_fc2_params = torch.cat([x.view(-1) for x in net_randomWts.fc2.parameters()])
        all_fc3_params = torch.cat([x.view(-1) for x in net_randomWts.fc3.parameters()])
        # The regularization term is the parameter lambda_1 * the norm of
        # the weights (will be inducing sparsity in the weights).
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


#==== VALIDATING (TUNING) THE MODEL ===========================================
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
    
# Print the validation accuracy of the network for the given set of 
# hyperparameters.
print("Validation accuracy: ", round(100*(correct.item()/total), 3), "%", sep="")


#==== TRAINING THE MODEL, PART II =============================================
# After having used validation to tune hyperparameters, go back and train on
# the original training data (to have 10,000 additional data points).
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
        
        #== DEFINE L1 REGULARIZATION ==========================================
        # The "all_fc_params" contain all the weights for the respective layers
        # (i.e. all_fc1_params contains all weights connecting input w/hidden1)
        all_fc1_params = torch.cat([x.view(-1) for x in net_randomWts.fc1.parameters()])
        all_fc2_params = torch.cat([x.view(-1) for x in net_randomWts.fc2.parameters()])
        all_fc3_params = torch.cat([x.view(-1) for x in net_randomWts.fc3.parameters()])
        # The regularization term is the parameter lambda_1 * the norm of
        # the weights (will be inducing sparsity in the weights).
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


#==== TESTING THE MODEL =======================================================
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





















