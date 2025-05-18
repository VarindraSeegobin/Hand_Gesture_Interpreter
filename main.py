from model import FingerCNN
from DataExtractor import FingersDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Dataset and DataLoader
train_dataset = FingersDataset(os.getcwd() + "/Fingers-Numbers-7/train/images", os.getcwd() + "/Fingers-Numbers-7/train/labels")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = FingerCNN(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.long().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'finger_model.pth')

