import torch
from torchvision import transforms
from torchvision.datasets import FGVCAircraft
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


#Selecting the model we want to test
MODEL_TYPE = "layer4"   # "fc" or "layer4"

#Step 1: Adding transform. To get all images to have a consistent size. (same as in Train.py)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet means
        std=[0.229, 0.224, 0.225]     # ImageNet stds
    ),
])

#Step 2: load dataset

test_dataset = FGVCAircraft( root="data", split="val", download=False, transform=transform )

#information about Batch sizes and number of batches

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_batches = len(test_loader)

print("Number of test batches:", num_batches)
print("Test samples:", len(test_dataset))

#Step 3: Load ResNet18 architecture and trained weights

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Recreating the model architecture
model = models.resnet18(weights=None)   # IMPORTANT: weights=None for testing, different from Train.py for not overwriting it

# Replace final layer to match training
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)

# Load trained weights (choose the model to display from the top)
if MODEL_TYPE == "fc":
    model.load_state_dict(torch.load("best_model_fc.pth", map_location=device))
elif MODEL_TYPE == "layer4":
    model.load_state_dict(torch.load("best_model_layer4.pth", map_location=device))
else:
    raise ValueError("Invalid MODEL_TYPE")

# Move to device and set eval mode
model = model.to(device)
model.eval()

#Step 4: Test/Evaluation

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

#Step 5: visualizing the models predictions

# Get one batch from test loader
batch_idx = 46   # change this based on which batch number we want to visualize

for i, (images, labels) in enumerate(test_loader):
    if i == batch_idx:
        break

images = images.to(device)
labels = labels.to(device)

# Forward pass
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

#Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

#Normalizing confusion matrix for readability
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_normalized,
    cmap="Blues",
    xticklabels=False,
    yticklabels=False
)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#Step 6: Plotting

# Move tensors to CPU for plotting
images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()

# Plot
plt.figure(figsize=(12, 12))

for i in range(9):
    plt.subplot(3, 3, i + 1)

    img = images[i].permute(1, 2, 0)

    # UNNORMALIZE (important!)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std + mean
    img = img.clamp(0, 1)

    plt.imshow(img)
    plt.axis("off")

    true_label = test_dataset.classes[labels[i]]
    pred_label = test_dataset.classes[preds[i]]

    color = "green" if preds[i] == labels[i] else "red"
    plt.title(f"P: {pred_label}\nT: {true_label}", color=color)

plt.tight_layout()
plt.show()