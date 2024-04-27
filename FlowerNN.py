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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
pool = nn.MaxPool2d(kernel_size=2, stride=2)


# Uncomment to figure out the in_features number for the first Linear layer
# for i, (images, labels) in enumerate(train_loader):
# 	x = conv1(images)
# 	x = pool(x)
# 	x = conv2(x)
# 	x = pool(x)
# 	print(x.shape)
# 	break

class FlowerNN(nn.Module):
	def __init__(self, activation_fn = F.relu):
		super().__init__()
		self.conv1 = conv1
		self.pool = pool
		self.conv2 = conv2
		self.fc1 = nn.Linear(in_features=32 * 54 * 54, out_features=136)
		self.fc2 = nn.Linear(136, 114)
		self.fc3 = nn.Linear(114, 102)

		self.activation_fn = activation_fn

	def forward(self, x):
		x = self.pool(self.activation_fn(self.conv1(x)))
		x = self.pool(self.activation_fn(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = self.activation_fn(self.fc1(x))
		x = self.activation_fn(self.fc2(x))
		x = self.fc3(x)
		return x
	
classifier = FlowerNN()
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
epochs = 10

losses = []
for i in range(epochs):
	for images, labels in train_loader:
		optimizer.zero_grad()
		y_pred = classifier.forward(images)
		loss = lossFn(y_pred, labels)
		losses.append(loss.detach().numpy())
		loss.backward() 
		optimizer.step()
		
	print(f"Epoch {i} - {loss}")

# # Evaluate
# with torch.no_grad(): # Tell pytorch not to calculate the gradient
# 	y_eval = classifier.forward(x_validate)
# 	loss = lossFn(y_eval, y_validate)

# # # Visualize the loss for each epoch and the final loss of the trained model
# import matplotlib.pyplot as plt
# plt.plot(range(epochs), losses) # The loss for each epoch
# plt.plot([epochs], [loss], "g+") # The final loss
# plt.ylabel("Loss")
# plt.xlabel("Epoch")