import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from mycnn import SimpleCNN
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

full_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# ------------------
# Device
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(f"Using device: {device}")
#quit()

# ------------------
# Model, loss, optimizer
# ------------------
model = SimpleCNN(num_classes=10).to(device)

#dislay model parameters
#print(list(model.parameters()))
#quit()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# ------------------
# Training loop
# ------------------

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stats
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    # ------------------
    # Validation
    # ------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    scheduler.step()

# ==================
# Test evaluation
# ==================

# Load best model weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Test dataset and loader (no augmentation, same as validation)
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

correct_top1 = 0
total = 0
correct_top3 = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Top-1 accuracy
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()

        # Top-3 accuracy
        _, top3 = outputs.topk(3, dim=1)
        correct_top3 += top3.eq(labels.view(-1, 1)).sum().item()

        total += labels.size(0)


top1_acc = 100. * correct_top1 / total
top3_acc = 100. * correct_top3 / total

print("\nTest set evaluation")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-3 Accuracy: {top3_acc:.2f}%")