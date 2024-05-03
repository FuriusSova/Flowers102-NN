import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
import matplotlib.pyplot as plt

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
	v2.ToDtype(torch.float32, scale=True),
	v2.RandomHorizontalFlip(),
	v2.RandomRotation(15),
    v2.Normalize(
        mean=[0.4330, 0.3819, 0.2964],
        std=[0.2587, 0.2093, 0.2210]
    ) 
])

train_set = datasets.Flowers102(root='./train', split='train', download=True, transform=transform)
val_set = datasets.Flowers102(root='./valid', split='val', download=True, transform=transform)
test_set = datasets.Flowers102(root='./test', split='test', download=True, transform=transform)

batch_size = round(len(train_set) / 12)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
# conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
# conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

# Uncomment to figure out the in_features number for the first Linear layer
# for i, (images, labels) in enumerate(train_loader):
# 	x = conv1(images)
# 	x = pool1(x)
# 	x = conv2(x)
# 	x = pool2(x)
# 	# x = conv3(x)
# 	# x = conv4(x)
# 	# x = pool2(x)
# 	print(x.shape)
# 	break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerNN(nn.Module):
	def __init__(self, activation_fn = F.relu):
		super().__init__()
		self.conv1 = conv1
		self.conv2 = conv2
		# self.conv3 = conv3
		# self.conv4 = conv4
		self.pool1 = pool1
		self.pool2 = pool2
		self.dropout = nn.Dropout(p=0.3)
		self.fc1 = nn.Linear(in_features=16 * 27 * 27, out_features=128)
		# self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(128, 102)

		self.activation_fn = activation_fn

	def forward(self, x):
		x = self.pool1(self.activation_fn(self.conv1(x)))
		x = self.pool2(self.activation_fn(self.conv2(x)))
		# x = self.activation_fn(self.conv3(x))
		# x = self.pool2(self.activation_fn(self.conv4(x)))
		x = torch.flatten(x, 1)
		x = self.activation_fn(self.fc1(x))
		# x = self.activation_fn(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)
		return x
	
@torch.no_grad()
def validate():
	classifier.eval()
	total_samples = 0
	total_correct = 0
	total_loss = 0
	for images, labels in val_loader:
		images = images.to(device)
		labels = labels.to(device)
		y_pred = classifier.forward(images)
		_, predicted = torch.max(y_pred, 1)
		val_loss = lossFn(y_pred, labels)
		# val_losses.append(val_loss.detach().numpy())
		total_loss += val_loss.item() * images.size(0)
		total_samples += labels.size(0)
		total_correct += (predicted == labels).sum().item()

	return {"val_loss": total_loss / total_samples, "val_acc": total_correct / total_samples}

	
classifier = FlowerNN().to(device)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.01)
epochs = 40

# train_losses = []
# val_losses = []
last_val_loss = 20
counter_neg_loss = 0
for i in range(epochs):
	classifier.train()
	for images, labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		y_pred = classifier.forward(images)
		loss = lossFn(y_pred, labels)
		# train_losses.append(loss.detach().numpy())
		loss.backward() 
		optimizer.step()
		
	val_results = validate()
	val_loss = val_results["val_loss"]
	val_acc = val_results["val_acc"]
	print(f"Epoch {i} - train loss: {loss}, val loss: {val_loss}, val_acc: {val_acc}")
	
	if last_val_loss < val_loss: 
		counter_neg_loss += 1
	else:
		counter_neg_loss -= 1
	
	if counter_neg_loss == 5:
		print("Validation loss is increasing")
		break
	last_val_loss = val_loss