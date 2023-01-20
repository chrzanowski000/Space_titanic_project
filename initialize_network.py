import torch
from torch import nn  # All neural network modules
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
"""
# Network prototype
class model_block(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(model_block, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        return y

class Net(torch.nn.Module):
    def __init__(self, D_in1, H1, D_out1, H2, D_out2):
        super(Net, self).__init__()
        self.block1 = model_block(D_in1, H1, D_out1).double()
        self.block2 = model_block(D_out1, H2, D_out2).double()

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        return torch.sigmoid(y)
    """
# create class
class Net(nn.Module):
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(Net,self).__init__()
        # Linear function.
        self.linear1 = nn.Linear(input_size,100)
        self.linear2 = nn.Linear(100,100)
        self.linear3 = nn.Linear(100,100)
        self.linear4 = nn.Linear(100,output_size)

    def forward(self,x):
        y = self.linear1(x)
        y = self.linear2(y)
        y = self.linear3(y)
        y = self.linear4(y)
        return torch.sigmoid(y)