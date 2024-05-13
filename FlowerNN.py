import torch
import torch.nn as nn
import torch.nn.functional as F


conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

class FlowerNN(nn.Module):
	def __init__(self, activation_fn = F.relu):
		super().__init__()
		self.conv1 = conv1
		self.conv2 = conv2
		self.conv3 = conv3
		self.conv4 = conv4
		self.pool2 = pool2
		self.dropout3 = nn.Dropout(p=0.4)
		self.bn1 = nn.BatchNorm2d(128)
		self.bn2 = nn.BatchNorm2d(256)
		self.bn3 = nn.BatchNorm2d(512)
		self.fc1 = nn.Linear(in_features=512 * 3 * 3, out_features=1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 102)

		self.activation_fn = activation_fn

	def forward(self, x):
		x = self.activation_fn(self.conv1(x))
		x = self.bn1(self.pool2(self.activation_fn(self.conv2(x))))
		x = self.bn2(self.activation_fn(self.conv3(x)))
		x = self.bn3(self.pool2(self.activation_fn(self.conv4(x))))
		x = torch.flatten(x, 1)
		x = self.activation_fn(self.fc1(x))
		x = self.dropout3(x)
		x = self.activation_fn(self.fc2(x))
		x = self.dropout3(x)
		x = self.fc3(x)
		return x
	
