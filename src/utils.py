from random import random
import yaml #pip install pyyaml
import torch
import numpy as np
import random
from attrdict import AttrDict #pip install attrdict
from models.fc import *
from models.cnn import *

def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    return device

def set_seeds(seed=42):
    torch.backends.cudnn.enabled = False #don't let cuDNN use non-deterministic behavior
    torch.manual_seed(seed)  #for pytorch
    np.random.seed(seed)     #for numpy
    random.seed(seed)        #for python
    print("seed: ", seed)
    
def load_yaml(config_dir, filename):
    with open(config_dir / filename, "r") as stream:
        config = AttrDict(yaml.safe_load(stream))  #allow us to access the dict as both keys and attributes
    print("config: ", filename)
    return config

def load_model(config):
    class_name = list(config.keys())[0] #e.g., FC
    kwargs = list(config.items())[0][1] #e.g., input_size
    model_class = globals()[class_name] #get class from string
    
    print("model: ", class_name)

    model = model_class(**kwargs)       #make class instance
    
    return model