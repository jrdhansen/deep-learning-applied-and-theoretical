'''
    File name: hw3_prob1.py
    Author: Jared Hansen
    Date created: 02/26/2019
    Date last modified: 02/26/2019
    Python Version: 3.6.4
'''

'''
CODE FOR HW4, PROLEM 3 IN [DEEP LEARNING: THEORY AND APPLICATIONS]
===============================================================================


PROBLEM 3.1
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
        
        
        
PROBLEM 3.2
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


"""
NOTE:
    I used Pytorch (both for CPU and GPU) as the framework for training and
    testing my neural network.
    
    For completion of this homework prompt, I used code from this tutorial as
    my starting point.
    https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c
    Here is how I changed his code:
        -- the architectures of the networks are different. As specified in the 
           prompt, our network is only to have a 784-node input layer and a
           10-node output layer. The architecture shown in the tutorial also
           has a 500-neuron hidden layer.
        -- I also wrote code that uses the GPU for computation. This took a lot
           of Googling and additional work over just running the code on CPU.
        -- I tried to make helpful comments to exhibit that I actually know
           what the code is doing.
"""





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
    
    def __init__(self, input_size, hid_size_1, hid_size_2, num_classes):
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
        # First/only fully-connected layer: 784 (input) --> 10 (output)
        self.fc1 = nn.Linear(input_size, num_classes)  
        # 
        self.relu = nn.ReLU()
        # Use sigmoid activation function to determine 10-node layer outputs.
        self.Sigmoid = nn.Sigmoid()
    
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
        out = self.fc1(x)
        # The output of the final layer after being passed through the Sigmoid
        # activation function.
        out = self.Sigmoid(out)
        # Returns the outputs of the final 10-node layer.
        return out
    
    
    
    
    
#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Initialize hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 10
batch_size = 20
learning_rate = 1.0

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
net = Net(input_size, num_classes)

# Define a loss function and optimizer. The prompt didn't specify a loss 
# function, so I'm using cross-entropy (commonly used loss function for nets).
# The prompt did specify to use stochastic gradient descent, hence the SGD
# optimizer. I left the momentum argument as the default of 0.0.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate,
                            momentum=0.0)

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
        outputs = net(images)
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
    outputs = net(images)
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
#==== Using GPU ===============================================================
#==============================================================================

#==== IS A PYTORCH-ENABLED GPU AVAILABLE? =====================================
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()  # if(output is > 0) ==> there is a CUDA device 
                           # that Pytorch can use.
torch.cuda.get_device_name(0)  # Will give the name of the CUDA device
                               # Mine is 'GeForce 940MX'
                               
                               
# Using the .to(device) command and the .cuda() command does the exact same
# thing (tells the interpreter to use the GPU device for computing).

# Necessary .to(device) commands if using .to(device) instead of .cuda
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
#images, labels = images.to(device), labels.to(device)

# In order to use the GPU we need to tell the interpreter to make other objects
# CUDA-compatible (that's my vague understanding of what we're doing here).
images, labels = images.cuda(), labels.cuda(async=True)

# Instantiate the network and make it GPU-compatible. Then define the loss 
# function (also needs to be GPU-compatible).
net = Net(input_size, num_classes)
net = net.cuda()
criterion = nn.CrossEntropyLoss().cuda()
# Set the optimizer to be stochastic gradient descent.
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.0)
#optimizer = optim.SGD(net.parameters(), lr=learning_rate)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

#==== Training the model ======================================================
# We'll iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(train_loader):
        # Convert the input images to a torch tensor that is GPU-enabled.
        images = Variable(torch.FloatTensor(images.view(-1,28*28)).cuda())
        # Also have the labels tensor be GPU-enabled.
        labels = Variable(labels).cuda()
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network.
        outputs = net(images)
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
    # 28x28 matrix, GPU-enabled.
    images = Variable(images.view(-1, 28*28).cuda())
    # Determine predicted output (0,1,2,...,9) for each image in the test data
    # by passing them through the trained network.
    outputs = net(images)
    # Determine the predicted label for this image by choosing the node with
    # the highest (max) activation in the output layer.
    _, predicted = torch.max(outputs.data, 1)
    # Increment the total variable by 1 each time through the loop (total is a
    # simple int object).
    total += labels.size(0)
    # Increment correct by 1 each time through the loop if "predicted" is the
    # same as the true "labels" for that image. (This is a torch.Tensor.)
    correct += (predicted == labels.cuda()).sum()
    
# Print the test accuracy of the network.
print("Test accuracy: ", round(100*(correct.item()/total), 3), "%", sep="")
