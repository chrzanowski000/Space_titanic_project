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
        self.linear1 = nn.Linear(input_size,40)
        self.linear2 = nn.Linear(40,80)
        #self.linear3 = nn.Linear(500,100)
        self.linear3 = nn.Linear(80,output_size)
        self.norm10 = torch.nn.LayerNorm([10])
        self.norm50 = torch.nn.LayerNorm([30])

    def forward(self,x):

        """        
        y = F.leaky_relu(self.linear1(x), 0.2)
        y = F.leaky_relu(self.linear2(y), 0.2)
        y = F.leaky_relu(self.linear3(y), 0.2)
        y = torch.sigmoid(self.linear4(y))
        """
        
        y = F.leaky_relu(self.linear1(x))
        y = F.leaky_relu(self.linear2(y))
        #y = F.relu(self.linear3(y))
        y = torch.sigmoid(self.linear3(y))
        
        """
        y = self.norm50(torch.tanh(self.linear1(x)))
        y = torch.sigmoid(self.linear2(y))
        y = self.norm10(torch.tanh(self.linear3(y)))
        y = torch.sigmoid(self.linear4(y))
        """

        return y