import numpy as np
from typing import *

def dec2bin(num):
    l = []
    if num < 0:
        return '-' + dec2bin(abs(num))
    while True:
        num, remainder = divmod(num, 2)
        l.append( remainder )
        if num == 0:
            return l[::-1]

def bin2dec(bs):
    num = 0
    base = 2
    for i,b in enumerate(bs[::-1]):
        num += b*(base**i)
    return int(num)


def nums2onehotcode(nums:List,num_class,smooth_eps):
    one_hot_code = np.zeros((len(nums), num_class), np.float32) + smooth_eps / (num_class - 1)
    for idx,num in enumerate(nums):
        one_hot_code[idx, num] = 1.0 - smooth_eps
    return one_hot_code

def nums2binarycode(nums:List,binary_code_len):
    binay_code = np.zeros((len(nums), binary_code_len), np.float32)
    for idx,num in enumerate(nums):
        binary_code_list = dec2bin(num)
        assert len(binary_code_list) <= binary_code_len
        for j, v in enumerate(binary_code_list):
            binay_code[idx, binary_code_len - len(binary_code_list) + j] = v
    return binay_code

def out_dim(num_class:int) -> int:
    return int( np.log2(num_class - 1)) + 1