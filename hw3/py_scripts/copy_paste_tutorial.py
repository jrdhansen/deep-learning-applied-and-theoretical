

# MESS AROUND WITH GETTING gpu TO WORK IN THIS SCRIPT




# Borrowed stuff from this tutorial: https://towardsdatascience.com/@jj1385jeff850527
# Here is his github: https://github.com/yhuag/neural-network-lab/blob/master/Feedforward%20Neural%20Network.ipynb




#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable








#==============================================================================
#==== IS MY GPU PLAYING NICELY WITH PYTORCH? ==================================
#==============================================================================

torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)











#==============================================================================
#==== FUNCTIONS ===============================================================
#==============================================================================

class Net(nn.Module):
    #def __init__(self, input_size, hidden_size, num_classes):
     #   super(Net, self).__init__()                    # Inherited from the parent class nn.Module
     #   self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
     #   self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
     #   self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
    
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, num_classes)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.Sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        #self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
    
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.Sigmoid(out)
        #out = self.relu(out)
        #out = self.fc2(out)
        return out




#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Initialize hyper-parameters
input_size = 784
#hidden_size = 500
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
net = Net(input_size, num_classes)
#net = Net(input_size, hidden_size, num_classes)
# Enable GPU -- THIS DOES NOT WORK ----------------------------------------
#net.cuda()

# Choose a loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate)

# Training the FNN model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        #images = Variable(torch.FloatTensor(images.view(-1,28*28)).cuda())
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



#==============================================================================
#==== USING GPU ===============================================================
#==============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
#images, labels = images.to(device), labels.to(device)
images, labels = images.cuda(), labels.cuda(async=True)
net = Net(input_size, hidden_size, num_classes)
net = net.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.SGD(net.parameters(), lr=learning_rate)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Training the FNN model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        #images = Variable(torch.FloatTensor(images.view(-1,28*28)).cuda())
        images = Variable(images.view(-1, 28*28).to(device))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = Variable(labels).to(device)
        
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
    images = Variable(images.view(-1, 28*28).cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
    total += labels.size(0)                    # Increment the total count
    correct += (predicted == labels.cuda()).sum()     # Increment the correct count
    
print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))          






# Save the trained FNN model for future use
# torch.save(net.state_dict(), 'fnn_model.pkl')

