# Evaluate the model on test dataset
from FlowerNN import FlowerNN
from sklearn import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

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

directory = '/tmp/{university username}/train'

test_set = datasets.Flowers102(root=directory, split='test', download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device.type} device is in use")

trained_model = FlowerNN().to(device)
trained_model.load_state_dict(torch.load("./trained_model.pt"))
lossFn = nn.CrossEntropyLoss()

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