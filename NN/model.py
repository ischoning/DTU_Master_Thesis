import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, LSTM, GRU, Conv1d, Conv2d, Dropout, MaxPool2d, BatchNorm1d, BatchNorm2d, CrossEntropyLoss, BCELoss
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from Constants import *

"""
Define the model
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_1 = Conv2d(in_channels = CHANNELS,
                             out_channels = OUT_CHANNELS[0],
                             kernel_size = KERNEL_SIZE,
                             stride = STRIDE,
                             padding = PADDING)

        self.conv_2 = Conv2d(in_channels = OUT_CHANNELS[0],
                             out_channels = OUT_CHANNELS[1],
                             kernel_size = KERNEL_SIZE[1],
                             stride = STRIDE,
                             padding = PADDING)
        
        self.lin_1 = Linear(in_features = 784,
                           out_features = 96,
                           bias=False)
        
        # Output linear layer
        self.l_out = Linear(in_features = 96,
                            out_features = NUM_CLASSES,
                            bias = False)

        # Functions to help generalize the model
        self.batchnorm2d_1 = BatchNorm2d(CHANNELS)
        self.batchnorm2d_2 = BatchNorm2d(OUT_CHANNELS[0])
        #self.maxp = MaxPool2d(kernel_size = (1,OUT_CHANNELS[0]), stride = (1,5))
        self.dropout = Dropout(p = 0.2)
        self.batchnorm2d_3 = BatchNorm2d(OUT_CHANNELS[1])
        
    def forward(self, x_img):
        out = {}
        features = []
        batch_size = x_img.shape[0]
        
        ## Convolutional layer ##
        # - Change dimensions to fit the convolutional layer 
        # - Apply Conv2d
        # - Use an activation function
        x_img = x_img.permute(0, 3, 1, 2)  # input in the shape (batch_size, in_channels, height, width)
        x_img = self.conv_1(x_img)
        x_img = relu(x_img)
        #x_img = self.maxp(x_img)
        x_img = self.batchnorm2d_2(x_img)
        x_img = self.dropout(x_img)
        
        #x_img = x_img.permute(0, 2, 1, 3)
        x_img = self.conv_2(x_img)
        x_img = relu(x_img)
        x_img = self.batchnorm2d_3(x_img)
        x_img = self.dropout(x_img)
        
        #x_img = self.batchnorm2d_3(x_img)
        features_img = x_img.reshape(batch_size, -1)
        
        # Linear layer
        features_img = self.lin_1(features_img)
        
        ## Output layer ##
        out['out'] = softmax(self.l_out(features_img))
        
        return out
