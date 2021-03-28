import importlib
from train_eval import train_eval

cfg = importlib.import_module('config.cifar10')
train_eval(cfg)