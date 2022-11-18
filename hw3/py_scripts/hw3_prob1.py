'''
    File name: hw3_prob1.py
    Author: Jared Hansen
    Date created: 02/05/2019
    Date last modified: 02/05/2019
    Python Version: 3.6.4
'''


'''
CODE FOR HW3, PROLEM 1.3 IN [DEEP LEARNING: THEORY AND APPLICATIONS]

PROMPT: Create a network that classifies the MNIST data set using only 2
        layers: the input layer (784 nuerons) and the output layer (10
        neurons). Train the network using stochastic gradient descent on the 
        training data. What accuracy do you achieve on the test data? You can
        adapt the code from the Nielson book, but make sure you understand each
        step to build up the network. Alternatively, you can use Tensorflow or
        Pytorch.
        Please save your code as prob1.py and state which library/framework you
        used. Report the learning rate(s) and mini-batch size(s) you used.
'''





#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


















#==============================================================================
#==== FUNCTIONS ===============================================================
#==============================================================================

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out




#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================


# Initialize hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# Download the MNIST data
train_dataset = dsets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Load the MNIST data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Instantiate the Feed-forward NN
net = Net(input_size, hidden_size, num_classes)
# Enable GPU -- THIS DOES NOT WORK ----------------------------------------
#net.cuda()

# Choose a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training the FNN model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        images = Variable(images.view(-1, 28*28))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = Variable(labels)
        
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        outputs = net(images)                             # Forward pass: compute the output class given a image
        loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes
        
        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

# Testing the FNN model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
    total += labels.size(0)                    # Increment the total count
    correct += (predicted == labels).sum()     # Increment the correct count
    
print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))          

# Save the trained FNN model for future use
#torch.save(net.state_dict(), "fnn_model.pkl")

