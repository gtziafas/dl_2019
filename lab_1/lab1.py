import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models  
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split

import learning_loop as ll

#use gpu tensors if available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print('Using {}.\n'.format(device))

#define the model architectures with random initialized weights
alexnet = models.alexnet()
vgg16 = models.vgg16()
inception = models.inception_v3()
resnet18 = models.resnet18()
squeezenet = model.squeezenet1_0()
googlenet = models.googlenet()

