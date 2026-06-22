from torchvision.datasets import FGVCAircraft
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import json

TRAIN_MODEL = False   # We will turn this to true if we want to train again.
EPOCHS = 100
#Step 1: Adding transform. To get all images to have a consistent size.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet means
        std=[0.229, 0.224, 0.225]     # ImageNet stds
    ),
])

#Step 2: Load Datasets

train_dataset = FGVCAircraft(
    root="data",
    split="train",
    download=False,
    transform=transform
)

#Information about Train sample and Classes
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Train samples:", len(train_dataset))
print("Classes:", len(train_dataset.classes))

# Validation dataset
val_dataset = FGVCAircraft(
    root="data",
    split="val",
    download=False,
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Validation samples:", len(val_dataset))

#Step 4: Load Pretrained RESNET18

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replaces final layer to match 100 aircraft classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)

# Make the final layer(fc) and the layer before it (layer4) trainable

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

#Step 5: Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

#Step 6: Training Loop (We modified it so that it runs only when Train_Model = True)

train_losses = []
val_losses = [] #Loss validation
best_val_loss = float("inf") #variable for early stopping
patience = 10
patience_counter = 0

if TRAIN_MODEL:
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save BEST model
            torch.save(model.state_dict(), "best_model_fc.pth")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save model and losses
    torch.save(model.state_dict(), "resnet18_aircraft_fc.pth")

    with open("train_losses_fc.json", "w") as f:
        json.dump(train_losses, f)

    # Save Loss Curves (Train + Validation)

    with open("val_losses_fc.json", "w") as f:
        json.dump(val_losses, f)


#This loads the saved model when we don't retrain
if not TRAIN_MODEL and os.path.exists("best_model_fc.pth"):
    model.load_state_dict(torch.load("best_model_fc.pth", map_location=device))
    model.eval()

#Step 7: Plotting Loss Curve after Training

#Loads the saved losses so that we don't have to retrain everytime
if os.path.exists("train_losses_fc.json"):
    with open("train_losses_fc.json", "r") as f:
        train_losses = json.load(f)

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epochs (Fc-only)")
    plt.grid(True)
    plt.show()
else:
    print("No training loss file found. Train the model first.")

#Plotting Validation Loss

if os.path.exists("val_losses_fc.json"):
    with open("val_losses_fc.json", "r") as f:
        val_losses = json.load(f)

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (Fc-only)")
    plt.legend()
    plt.grid(True)
    plt.show()


