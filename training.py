from FlowerNN import FlowerNN
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

directory = '/tmp/{university usarname}/train'

train_set = datasets.Flowers102(root=directory, split='train', download=True, transform=transform)
val_set = datasets.Flowers102(root=directory, split='val', download=True, transform=transform)
test_set = datasets.Flowers102(root=directory, split='test', download=True, transform=transform)

batch_size = 16

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device.type} device is in use")

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

total_samples_test = 0
total_correct_test = 0
total_loss_test = 0
for images, labels in test_loader:
	classifier.eval()
	images = images.to(device)
	labels = labels.to(device)
	y_pred = classifier.forward(images)
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