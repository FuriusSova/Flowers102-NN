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

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

# dataiter = iter(train_loader)
# images, labels = next(dataiter)

def split_dataset(set_to_split):
	x_set = []
	y_set = []
	for image, label in set_to_split:
		# red_channel = image[0]
		# green_channel = image[1]
		# blue_channel = image[2]
		# red_channel = image[0].flatten().numpy()
		# green_channel = image[1].flatten().numpy()
		# blue_channel = image[2].flatten().numpy()
		x_set.append(image)
		y_set.append(label)
	
	return (x_set, y_set)

x_train, y_train = split_dataset(train_set)
x_validate, y_validate = split_dataset(val_set)
x_test, y_test = split_dataset(test_set)

x_train = torch.stack(x_train)
x_validate = torch.stack(x_validate)
x_test = torch.stack(x_test)

y_train = torch.LongTensor(y_train)
y_validate = torch.LongTensor(y_validate)
y_test = torch.LongTensor(y_test)

# df = pd.DataFrame(data, columns=['Red', 'Green', 'Blue', 'Label'])
# features = df[['Red', 'Green', 'Blue']].values
# labels = df['Label'].values

# features_tensor = torch.tensor(features, dtype=torch.float32)
# labels_tensor = torch.tensor(labels, dtype=torch.long)

# dataset = TensorDataset(features_tensor, labels_tensor)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

class FlowerNN(nn.Module):
	def __init__(self, in_features, hidden_layers, out_features, activation_function = F.relu): 
		super().__init__()
		if len(hidden_layers) < 1:
			raise Exception("My_NN must have at least 1 hidden layer")

		self.layers = []  
		self.layers.append(nn.Linear(in_features, hidden_layers[0])) 
		self.add_module("input_layer", self.layers[0]) 

		for i in range(1, len(hidden_layers)):
			self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
			self.add_module(f"hidden_layer_{i}", self.layers[i])

		self.out = nn.Linear(hidden_layers[-1], out_features)

		self.activation_function = activation_function

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activation_function(self.layers[i](x))
		x = self.out(x)
		return x
	
classifier = FlowerNN(in_features=224, hidden_layers=[128, 64], out_features=102, activation_function = F.relu)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
epochs = 500

losses = []
for i in range(epochs):
	y_pred = classifier.forward(x_train)
	y_pred = y_pred.view(1020, -1)
	loss = lossFn(y_pred, y_train)
	losses.append(loss.detach().numpy())

	if i % 10 == 0:
		print(f"Epoch {i} - {loss}") # Print the loss every 10 epoch

	# Do corrections to thetas
	optimizer.zero_grad() # Clean the stored grads computed in the last iteration
	loss.backward() # Calculates the amounts given the differentials. How to correct each of the neurons given the loss that we have
	optimizer.step()

# Evaluate
with torch.no_grad(): # Tell pytorch not to calculate the gradient
	y_eval = classifier.forward(x_validate)
	loss = lossFn(y_eval, y_validate)

# # Visualize the loss for each epoch and the final loss of the trained model
import matplotlib.pyplot as plt
plt.plot(range(epochs), losses) # The loss for each epoch
plt.plot([epochs], [loss], "g+") # The final loss
plt.ylabel("Loss")
plt.xlabel("Epoch")