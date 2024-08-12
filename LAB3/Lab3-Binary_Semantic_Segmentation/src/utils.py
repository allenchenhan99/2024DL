import torch

def dice_score(pred_mask, gt_mask):
    
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()
    
    intersection = (pred_mask * gt_mask).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    # print(intersection)
    # print(union)
    dice = (2. * intersection) / (union)
    # print(dice.shape)

    avg_dice = torch.mean(dice)
    return avg_dice.item()