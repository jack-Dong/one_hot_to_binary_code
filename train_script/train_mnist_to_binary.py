import importlib
from train_eval import train_eval

cfg = importlib.import_module('config.mnist')

train_eval(cfg)



