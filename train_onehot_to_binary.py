import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.onehot_to_binary_model import Net
import numpy as np
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)

bs_size = 32
num_class = 32
eps = 0.0001
lr = 0.0001

num_train_iteration = 2000
num_test_iteration = 100


def get_ips_and_labes(batch_size,num_class,smooth_eps,device):
    binary_code_len = int(np.log2(num_class-1)) + 1
    nums = list(np.random.randint(0, num_class, batch_size))
    one_hot_codes = nums2onehotcode(nums, num_class, smooth_eps=smooth_eps)
    binay_label = nums2binarycode(nums, binary_code_len)
    x = torch.from_numpy(one_hot_codes)
    x = x.to(device)
    y = torch.from_numpy(binay_label)
    y = y.to(device)

    return x,y,nums

def main():

    model = Net(num_class)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for iter in range(num_train_iteration):
        print('============= Train ==================')

        x,y,nums = get_ips_and_labes(bs_size,num_class,eps,device)

        print('iter',iter)

        y_hat = model(x)
        loss = criterion(y_hat,y)

        loss.backward()
        optimizer.step()

        print('loss',loss.item())
        print('nums',nums)

        pred_nums = []
        y_hat_np = np.float32( y_hat.cpu().detach().numpy() > 0.5 )
        for b in range(bs_size):
            pred_nums.append( bin2dec(y_hat_np[b]) )

        print('pred_nums',pred_nums)

        acc = np.sum( np.array(nums,np.int) == np.array(pred_nums,np.int) ) / len(nums)
        print('acc',acc)

    total_acc = 0
    model.eval()
    for iter in range(num_test_iteration):
        print('============= Test ===================')
        x, y, nums = get_ips_and_labes(bs_size, num_class, eps,device)
        print('iter', iter)
        y_hat = model(x)
        loss  = criterion(y_hat, y)
        print('loss', loss.item())
        print('nums', nums)
        pred_nums = []
        y_hat_np = np.float32(y_hat.cpu().detach().numpy() > 0.5)
        for b in range(bs_size):
            pred_nums.append(bin2dec(y_hat_np[b]))

        print('pred_nums', pred_nums)

        acc = np.sum(np.array(nums, np.int) == np.array(pred_nums, np.int)) / len(nums)
        total_acc += acc
        print('acc', acc)

    print('total_acc',total_acc / num_test_iteration)


if __name__ == '__main__':

    main()

    # print(bin2dec([1,1,1]))+--\,;--[-[]][