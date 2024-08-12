import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import json

from dataloader import ButterflyMothLoader
from ResNet50 import ResNet50
from VGG19 import VGG19


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def evaluate(model, data_loader, device, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / total
    accuracy = 100 * correct / total
    return val_loss, accuracy


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
    return accuracy

    

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_items = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_items += labels.size(0)
    average_loss = total_loss / total_items
    train_accuracy = (total_correct / total_items) * 100
    
    return average_loss, train_accuracy


if __name__ == "__main__":
    
    # model = ResNet50(num_classes=100).to(device)
    model = VGG19(num_features=25088, num_classes=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    early_stopping = EarlyStopping(patience=60, delta=0.01)
    num_epoch=60
    
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

    train_dataset = ButterflyMothLoader(root='./dataset', mode='train', transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    eval_dataset = ButterflyMothLoader(root='./dataset', mode='valid', transform=valid_test_transform)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=64, shuffle=False)
    test_dataset = ButterflyMothLoader(root='./dataset', mode='test', transform=valid_test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    history = {
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    model_file = 'main2_saved_models/VGG19.pth'
    if os.path.isfile(model_file):
        print("Loading saved model parameters...")
        model.load_state_dict(torch.load(model_file))

        val_loss, val_accuracy = evaluate(model, eval_loader, device, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        test_accuracy = test(model, test_loader, device)
        print(f'Test Accuracy: {test_accuracy:.2f}%')
    else:
        print("No saved model parameters found, starting training...")
        for epoch in range(num_epoch):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = evaluate(model, eval_loader, device, criterion)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            print(f'Epoch [{epoch+1}/{num_epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                torch.save(model.state_dict(), model_file)
                print("Model parameters saved")
                break
            
        if not early_stopping.early_stop:
            torch.save(model.state_dict(), model_file)
            print("Model parameters saved")
        
        history_dir = './history'
        model_history_name = 'VGG1950'
        json_file_path = os.path.join(history_dir, f'{model_history_name}_history.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(history, json_file)

        test_accuracy = test(model, test_loader, device)
        print(f'Test Accuracy: {test_accuracy:.2f}%')