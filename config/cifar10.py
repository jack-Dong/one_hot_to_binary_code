from torchvision import datasets, transforms
from model.minist_model import Net
import torch
from torchvision.models import resnet18
from utils import out_dim
import torch.optim as optim
from model.cifar10_model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


data_train = datasets.CIFAR10(root='../../data',
                              transform=transform,
                              train=True)
data_test = datasets.CIFAR10(root="../../data/",
                           transform = transform,
                           train = False)

bs_size   = 128
num_class = 10
lr = 0.001
num_epochs = 100

# model = resnet18(pretrained=False,
#                  num_classes=out_dim(num_class)
#                  )

model = Net(num_class)

optimizer = optim.Adam(model.parameters(), lr=lr)