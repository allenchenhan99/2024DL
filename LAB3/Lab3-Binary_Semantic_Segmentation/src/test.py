from oxford_pet import OxfordPetDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    train_dataset = OxfordPetDataset(root=r'.\dataset\oxford-iiit-pet', mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    
    dataiter = iter(train_loader)
    data = next(dataiter)
    features, labels = data
    print(features, labels)