'''
    File name: DL_hw5_prob5.py
    Author: Jared Hansen
    Date created: 03/26/2019
    Date last modified: 03/26/2019
    Python Version: 3.6.4

===============================================================================
CODE FOR HW5, PROLEM 5 IN [DEEP LEARNING: THEORY AND APPLICATIONS]
===============================================================================
'''










'''
-------------------------------------------------------------------------------
---- PROBLEM 5 ----------------------------------------------------------------
-------------------------------------------------------------------------------

PART 2: 
        - Design and train a CNN with at least two convolutional layers, each
          followed by a max-pooling layer, for the MNIST dataset. 
        - Record your final test result and give a description on how you
          designed the network and briefly describe how you made those choices
          (e.g., numbers of layers, initialization strategies, parameter
          tuning, adaptive learning rate or not, momentum or not, etc.)
        - Based on the results you obtained, does the CNN seem to do better
          than other models you've trained?
        - If you're using Tensorflow, you may want to modify the network based
          on the following link, if you don't know what to start with. If
          you're using pyTorch, the following link may still be helfpul
          https://www.tensorflow.org/tutorials/layers        
        
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
    
class ConvNet(nn.Module):
    """
    This class intializes and specifies the structure for our neural network:
    -- two [conv+MP layers] followed by two fully-connected layers (100 nodes)
    The nn.Module is the base class for all neural network modules.
    """
    def __init__(self, channels_conv1, channels_conv2, hid_size_1, hid_size_2):
        """
        This function initializes the neural network (constructor). We specify
        the [conv+MP] layer sizes, filter sizes and strides, nodes in fc layers
        and activation functions.
        
        Parameters
        ----------
        channels_conv1 : int
            This is the number of channels we create for the first 
            convolutional layer.
        channels_conv2 : int
            This is the number of channels we create for the second
            convolutional layer. 
        hid_size_1 : int
            This is the number of neurons in the first fully-connected layer.
        hid_size_2 : int
            This is the number of neurons in the second fully-connected layer.
            
        Returns
        -------
        n/a
        """
        
        # Inherited from the parent class nn.Module
        super(ConvNet, self).__init__()
        # First [conv+MaxPool] layer
        self.layer1 = nn.Sequential(
            # Specify the convolution: the images are 1-channel inputs, we 
            # choose how many channels to create. I left the filter size at 5x5
            # and the stride at 1 with padding of 2 around edges.
            nn.Conv2d(1, channels_conv1, kernel_size=5, stride=1, padding=2),
            # Used the ReLU activation since this is what we'd talkd about 
            # using during letures.
            nn.ReLU(),
            # Next, maxPool the conv layer. I left the filter size at 2x2 and
            # the stride at 2 (faster dimensionality reduction)
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Second [conv+MaxPool] layer, nearly identical to the first. The only
        # difference is that now we have as many channels as we created in the
        # first as our number of input channels, and specify a new number of 
        # output channels.
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels_conv1, channels_conv2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Implement dropout
        self.drop_out = nn.Dropout()
        # Create the first fully-connected layer. Since the output of the last
        # [conv+MP] layer is 7x7 (due to max-pooling downsampling) and we have
        # 64 channels, the number of input nodes is (7*7*64), and we specify
        # the number of nodes connected to is the hid_size_1 we specify.
        self.fc1 = nn.Linear(7 * 7 * 64, hid_size_1)
        # Then we simply connect the first fc-layer to the second fc-layer, 
        # with the number of inputs being the number of nodes in fc1 and the
        # number of nodes connected to his the hid_size_2 we specify.
        self.fc2 = nn.Linear(hid_size_1, hid_size_2)   
    
    def forward(self, x):
        """
        This function performs the forward passing of the network, stacking the
        layers on top (side-by-side) of each other, computing values and
        passing those values to the next layer.
        
        Parameters
        ----------
        x: tensor(?) of floats
            Not 100% sure what this object is in terms of its type. But I do 
            know that it is the input of the first layer (i.e. the original
            input pixel values for a given image).
            
        Returns
        -------
        out : tensor(?) of floats
            Similarly to x, I'm not sure if this is a tensor, matrix, or
            vector. But I do have some confidence in saying that this contains
            the predicted response values for a given input (single image) or
            set of images (mini-batch).
        """
        # Create the first [conv+MP] layer
        out = self.layer1(x)
        # Pass/connect the first [conv+MP] layer to the second [conv+MP] layer
        out = self.layer2(out)
        # Reshape the output of this layer 
        out = out.reshape(out.size(0), -1)
        # Implement  dropout
        out = self.drop_out(out)
        # Pass thru the first fully connected layer
        out = self.fc1(out)
        # Pass thru the second fully connected layer
        out = self.fc2(out)
        # Return the predicted output (either a vector, matrix, or tensor)
        return out


    
#==============================================================================
#==== PROCEDURAL CODE =========================================================
#==============================================================================

# Initialize hyper-parameters (fixed)
input_size = 784
num_classes = 10
num_epochs = 4

# Initialize hyper-parameters (tuned)
learning_rate = 0.001
batch_size = 128
hid_size_1 = 100
hid_size_2 = 100
channels_conv1 = 32
channels_conv2 = 64


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

# Instantiate the network, specifying architecture parameters (number of 
# channels for the [conv+MP] layers, and num of nodes in FC layers).
model = ConvNet(channels_conv1, channels_conv2, hid_size_1, hid_size_2)

# Specify the loss function (cross-entropy) and optimizer (Adam).
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




#==== TRAINING THE MODEL, PT I ================================================
#==============================================================================
# Train the model as part of hyperparameter tuning.
total_step = len(new_train_loader)
# Initialize the list to hold loss fctn values and accuracy for each step
loss_list = []
acc_list = []
# Iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(new_train_loader):
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network (model).
        outputs = model(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.        
        loss = criterion(outputs, labels)
        # Add the computed loss to the list of loss values.
        loss_list.append(loss.item())
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.        
        optimizer.step()
        # Update values for calculating accuracy (total images processed in
        # the current step/epoch)
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        # Determine which images in this batch were predicted correctly.
        correct = (predicted == labels).sum().item()
        # Compute the accuracy for this batch, append it to the list of acc's.
        acc_list.append(correct / total)
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))



#==== VALIDATING (TUNING) THE MODEL ===========================================
#==============================================================================
# This code specifies how we take the trained model and fit onto the validation
# set to select parameters.
model.eval()
with torch.no_grad():
    # Intiialize number of correctly-classified images, total number of images
    correct = 0
    total = 0
    # For each of the images in the validation set:
    for images, labels in val_loader:
        # Predict the output for that image
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # Increment the total number of test images by 1 each time thru loop
        total += labels.size(0)
        # Increment the correct number of predictions by 1 each time thru the
        # loop if the image is classified correctly by the trained model.
        correct += (predicted == labels).sum().item()
    # Print the validation accuracy on the 10000 validation images
    print('Validation Accuracy (on the 10000 validation images from original training data): {:.4f} %'.format((correct / total) * 100))



#==== TRAINING THE MODEL, PT II ===============================================
#==============================================================================
# After having used validation to tune hyperparameters, go back and train on
# all of the original training data (to have 10,000 additional data points).
total_step = len(train_loader)
# Initialize the list to hold loss fctn values and accuracy for each step
loss_list = []
acc_list = []
# Iterate over the data num_epochs times.
for epoch in range(num_epochs):
    # Load a batch of images and corresponding labels, index them with i.
    for i, (images, labels) in enumerate(train_loader):
        # Determine predicted output (0,1,2,...,9) for each image in this batch
        # by passing training data through the network (model).
        outputs = model(images)
        # Compute the (cross entropy) loss of the prediction VS the true label.        
        loss = criterion(outputs, labels)
        # Add the computed loss to the list of loss values.
        loss_list.append(loss.item())
        # Initialize all the weights to zero.
        optimizer.zero_grad()
        # Compute the updated weights based on the gradient of the loss fctn. 
        loss.backward()
        # Update the weights in the network using backpropogation.        
        optimizer.step()
        # Update values for calculating accuracy (total images processed in
        # the current step/epoch)
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        # Determine which images in this batch were predicted correctly.
        correct = (predicted == labels).sum().item()
        # Compute the accuracy for this batch, append it to the list of acc's.
        acc_list.append(correct / total)
        # This prints to the console to apprise us of the network's progress in
        # training, telling us which epoch and step we're on.
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))



#==== TESTING THE MODEL =======================================================
#==============================================================================
# Test the model
model.eval()
with torch.no_grad():
    # Intiialize number of correctly-classified images, total number of images
    correct = 0
    total = 0
    # For each of the images in the validation set:
    for images, labels in test_loader:
        # Predict the output for that image
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # Increment the total number of test images by 1 each time thru loop
        total += labels.size(0)
        # Increment the correct number of predictions by 1 each time thru the
        # loop if the image is classified correctly by the trained model.
        correct += (predicted == labels).sum().item()
    # Print the validation accuracy on the 10000 validation images
    print('Test Accuracy (on the 10000 test images): {:.4f} %'.format((correct / total) * 100))

# Save the model
MODEL_STORE_PATH = "C:/__JARED/_USU_Sp2019/_Moon_DeepLearning/hw5/"
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')


