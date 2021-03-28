import torch
import numpy as np
from utils import out_dim

class Net(torch.nn.Module):

    def __init__(self,num_class):
        num_output_hidden = out_dim(num_class)
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1 ,64 ,kernel_size=3 ,stride=1 ,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64 ,128 ,kernel_size=3 ,stride=1 ,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2 ,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 *14*128,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, num_output_hidden))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x