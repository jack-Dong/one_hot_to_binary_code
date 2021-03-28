import torch
import numpy as np
from utils import out_dim
from torch import nn

class Net(torch.nn.Module):

    def __init__(self,num_class):
        num_output_hidden = out_dim(num_class)
        super(Net, self).__init__()

        self.conv1 = torch.nn.Sequential(nn.Conv2d(3 ,64 ,kernel_size=3 ,stride=1 ,padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(64 ,128 ,kernel_size=3 ,stride=1 ,padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(stride=2 ,kernel_size=2))


        self.conv2 = torch.nn.Sequential(nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(stride=2, kernel_size=2))


        self.conv3 = torch.nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = torch.nn.Sequential(torch.nn.Linear(512,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, num_output_hidden))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x