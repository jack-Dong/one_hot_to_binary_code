import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):

    def __init__(self, num_class=1024):
        super(Net, self).__init__()

        num_output_hidden = int(np.log2(num_class - 1)) + 1

        self.fc1 = nn.Linear(num_class, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_output_hidden)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmod(x)
        return x


if __name__  == "__main__":

    print(Net())