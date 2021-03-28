from torchvision import datasets, transforms
from model.minist_model import Net
import torch
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])

data_train = datasets.MNIST(root = "../../data/",
                            transform=transform,
                            train = True,
                            download = False)

data_test = datasets.MNIST(root="../../data/",
                           transform = transform,
                           train = False)

bs_size   = 64
num_class = 10
lr = 0.0001
num_epochs = 10

model = Net(num_class)

optimizer = optim.Adam(model.parameters(), lr=lr)