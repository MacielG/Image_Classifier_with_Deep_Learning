import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    num_train = len(trainset)
    num_valid = int(0.1 * num_train)
    train_dataset, valid_dataset = random_split(trainset, [num_train - num_valid, num_valid])

    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    validloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2)

    net = SimpleCNN()

    model_path = 'cifar10_model.pth'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

    early_stopping_patience = 5
    min_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(10):
        net.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        net.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(validloader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), model_path)
            print("Model saved as the best model.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    print("Finished Training")

if __name__ == '__main__':
    main()
