import torchvision
import torch
from torchvision import datasets, transforms
from model.minist_model import Net
import numpy as np
from utils import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)

bs_size   = 64
num_class = 10
lr = 0.0001
num_epochs = 10


transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])

data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)


data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = bs_size,
                                                shuffle = True
                                                )

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = bs_size,
                                               shuffle = True)



model = Net(num_class)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, num_epochs))
    print("-" * 10)

    for i in tqdm(range(len(data_loader_train))):

        X_train, y_train = next(iter(data_loader_train))
        nums  = y_train.numpy()
        # print('nums',nums)

        X_train = X_train.to(device)
        y_train =  torch.from_numpy(nums2binarycode(nums, int(np.log2(num_class - 1)) + 1)).to(device)

        y_hat = model(X_train)

        y_hat_np = np.float32(y_hat.cpu().detach().numpy() > 0.5)
        pred_nums = []
        for b in range(y_hat_np.shape[0]):
            pred_nums.append(bin2dec(y_hat_np[b]))

        # print('pred_nums', pred_nums)

        optimizer.zero_grad()

        # print('y_hat',y_hat)
        # print('y_train',y_train)

        loss = criterion(y_hat, y_train)

        loss.backward()
        optimizer.step()
        loss_np = loss.item()
        # print('loss',loss_np)
        running_loss += loss_np

        correct = np.sum(np.array(nums, np.int) == np.array(pred_nums, np.int))
        batch_acc = correct / len(nums)
        # print('batch_acc', batch_acc)

        running_correct += correct

    testing_correct = 0

    model.eval()
    for data in data_loader_test:
        X_test, y_test = data
        nums    = y_test.numpy()
        X_test = X_test.to(device)
        y_test = torch.from_numpy(nums2binarycode(nums, int(np.log2(num_class - 1)) + 1)).to(device)
        y_hat = model(X_test)
        y_hat_np = np.float32(y_hat.cpu().detach().numpy() > 0.5)

        pred_nums = []
        for b in range(y_hat_np.shape[0]):
            pred_nums.append(bin2dec(y_hat_np[b]))

        # print('nums', nums)
        # print('pred_nums', pred_nums)

        correct = np.sum(np.array(nums, np.int) == np.array(pred_nums, np.int))
        batch_acc = correct / len(nums)
        # print('test_batch_acc', batch_acc)

        testing_correct += correct


    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                      100 * running_correct / len(
                                                                                          data_train),
                                                                                      100 * testing_correct / len(
                                                                                          data_test)))