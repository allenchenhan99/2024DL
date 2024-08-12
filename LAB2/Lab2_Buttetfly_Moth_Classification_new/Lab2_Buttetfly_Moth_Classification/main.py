import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

from dataloader import ButterflyMothLoader
from ResNet50 import ResNet50
from VGG19 import VGG19


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Train Accuracy: {accuracy:.2f}%')

def test(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    

def train(model, train_loader, criterion, optimizer, num_epoch=10):
    model.train()
    for epoch in range(num_epoch):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    
    model = ResNet50(num_classes=100).to(device)
    # model = VGG19(num_features=25088, num_classes=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print("Current device:", device)
    print("Model device:", next(model.parameters()).device)
    
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # resie 224 * 224
        transforms.RandomHorizontalFlip(),  # horizontal flipping
        transforms.RandomRotation(30),  # random rotating
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # color change
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # random crop
        transforms.ToTensor(),  # tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalization
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    model_file = 'saved_models/ResNet50(2)_lr=0.0001_batch=128_epoch=60'
    if os.path.isfile(model_file):
        print("Loading saved model parameters...")
        model.load_state_dict(torch.load(model_file))
    else:
        print("No saved model parameters found, starting training...")
        train_dataset = ButterflyMothLoader(root='./dataset', mode='train', transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        train(model, train_loader, criterion, optimizer, num_epoch=60)
        torch.save(model.state_dict(), model_file)
    
    eval_dataset = ButterflyMothLoader(root='./dataset', mode='valid', transform=valid_test_transform)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=128, shuffle=False)
    evaluate(model, eval_loader, device)
    
    test_dataset = ButterflyMothLoader(root='./dataset', mode='test', transform=valid_test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    test(model, test_loader, device)