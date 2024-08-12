import torch
from utils import dice_score

def evaluate(net, data, device):
    net.eval()
    total_dice = 0
    count = 0

    with torch.no_grad():
        for sample in data:
            images, true_masks, trimaps = sample["image"], sample["mask"], sample["trimap"]
            images, true_masks = images.to(device), true_masks.to(device)
            pred_mask = net(images)
            
            dice = dice_score(pred_mask, true_masks)
            total_dice += dice
            count += 1
    
    avg_dice = total_dice / count
    print(f"Average dice score: {avg_dice}")
    return avg_dice