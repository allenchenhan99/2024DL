import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet
from models.resnet34_unet import ResNet34Unet
from evaluate import evaluate
from oxford_pet import load_dataset

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    train_loader = load_dataset(data_path=args.data_path, mode='train', batch_size=args.batch_size)
    val_loader = load_dataset(data_path=args.data_path, mode='valid', batch_size=args.batch_size)
    
    # model = UNet(num_classes=1).to(device)
    model = ResNet34Unet(num_classes=1).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_dice = 0.0
    model.train()
    
    for epoch in range(args.epochs):
        for sample in train_loader:
            images, masks, trimaps = sample["image"], sample["mask"], sample["trimap"]
            images, masks, trimaps = images.to(device), masks.to(device), trimaps.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        avg_dice = evaluate(model, val_loader, device)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), '../saved_models/resnet34(4)_unet.pth')
            print(f'New best model saved with avg dice: {best_dice:.4f}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)