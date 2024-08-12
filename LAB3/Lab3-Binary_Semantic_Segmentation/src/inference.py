import argparse
import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34Unet
from oxford_pet import SimpleOxfordPetDataset
from oxford_pet import load_dataset
from utils import dice_score

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', type=int)
    parser.add_argument('--model_path', default='../saved_models/unet(5).pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    if args.model == 0:
        model = UNet(num_classes=1)
    else:
        model = ResNet34Unet(num_classes=1)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test_loader = load_dataset(data_path=args.data_path, mode='test', batch_size=args.batch_size)
    
    model.eval()
    total_dice = 0
    count = 0
    with torch.no_grad():
        for sample in test_loader:
            images, true_masks = sample['image'].to(device), sample['mask'].to(device)
            
            pred_masks = model(images)
            dice =  dice_score(pred_masks, true_masks)
            total_dice += dice
            count += 1
    avg_dice = total_dice / count
    print(f'Average Dice Score: {avg_dice}')