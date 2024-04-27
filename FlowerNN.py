import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(
        mean=[0.4330, 0.3819, 0.2964],
        std=[0.2587, 0.2093, 0.2210]
    ) 
])

train_set = shuffle(datasets.Flowers102(root='./train', split='train', download=True, transform=transform))
val_set = shuffle(datasets.Flowers102(root='./valid', split='val', download=True, transform=transform))
test_set = datasets.Flowers102(root='./test', split='test', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

