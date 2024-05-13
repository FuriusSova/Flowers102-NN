from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
	v2.RandomHorizontalFlip(),
	v2.RandomRotation(15),
	v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.4330, 0.3819, 0.2964],
        std=[0.2587, 0.2093, 0.2210]
    ) 
])

directory = '/tmp/ds1855/train'

train_set = datasets.Flowers102(root=directory, split='train', download=True, transform=transform)
val_set = datasets.Flowers102(root=directory, split='val', download=True, transform=transform)
test_set = datasets.Flowers102(root=directory, split='test', download=True, transform=transform)

batch_size = 16

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device.type} device is in use")

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
	
@torch.no_grad()
def validate():
	classifier.eval()
	total_samples_val = 0
	total_correct_val = 0
	total_loss_val = 0
	for images, labels in val_loader:
		images = images.to(device)
		labels = labels.to(device)
		y_pred = classifier.forward(images)
		_, predicted = torch.max(y_pred, 1)
		val_loss = lossFn(y_pred, labels)
		total_loss_val += val_loss.item() * images.size(0)
		total_samples_val += labels.size(0)
		total_correct_val += (predicted == labels).sum().item()

	return {"val_loss": total_loss_val / total_samples_val, "val_acc": total_correct_val / total_samples_val}


classifier = FlowerNN().to(device)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
epochs = 501

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
for i in range(epochs):
	classifier.train()
	total_samples_train = 0
	total_correct_train = 0
	total_loss_train = 0
	for images, labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		y_pred = classifier.forward(images)
		_, predicted = torch.max(y_pred, 1)
		loss = lossFn(y_pred, labels)
		total_loss_train += loss.item() * images.size(0)
		total_samples_train += labels.size(0)
		total_correct_train += (predicted == labels).sum().item()
		loss.backward() 
		optimizer.step()
		
	val_results = validate()
	val_loss = val_results["val_loss"]
	val_acc = val_results["val_acc"]

	# if val_loss < best_val_loss:
	# 	best_val_loss = val_loss
	# 	torch.save(classifier.state_dict(), "./trained_model.pt")

	train_losses.append(total_loss_train / total_samples_train)
	train_accuracies.append(total_correct_train / total_samples_train)
	val_losses.append(val_loss)
	val_accuracies.append(val_acc)

	scheduler.step()
	
	if i % 10 == 0:
		print(f"Epoch {i} - train loss: {loss}, val loss: {val_loss}, val_acc: {val_acc}, lr: {scheduler.get_last_lr()}")

# Evaluate the model on test dataset
trained_model = FlowerNN().to(device)
trained_model.load_state_dict(torch.load("./trained_model.pt"))

total_samples_test = 0
total_correct_test = 0
total_loss_test = 0
for images, labels in test_loader:
	trained_model.eval()
	images = images.to(device)
	labels = labels.to(device)
	y_pred = trained_model.forward(images)
	_, predicted = torch.max(y_pred, 1)
	val_loss = lossFn(y_pred, labels)
	total_loss_test += val_loss.item() * images.size(0)
	total_samples_test += labels.size(0)
	total_correct_test += (predicted == labels).sum().item()

print(f"Final test accuracy: {total_correct_test / total_samples_test}, test loss: {total_loss_test / total_samples_test}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./plot.png')