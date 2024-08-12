import os
from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import iclevrDataset

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from Unet import ContextUnet, NaiveUnet
from DDPM import DDPM

def train(n_epoch=300, device='cuda:0', load_path=None):
    # parameters
    batch_size = 32
    n_T = 1500
    device = "cuda:0"
    n_classes = 24
    n_feat = 256
    lr = 1e-4
    save_model = True
    save_dir = './model/diffusion_outputs10_lemb_tembV2_b32_NaiveUnetV6/'
    ws_test = [0.0, 0.5, 2.0]
    output_dir = './contents/batch_32_tembV2_lemb_NaiveUnetV6/'
    
    # the directory of storing images
    os.makedirs(output_dir, exist_ok=True)
    ddpm = DDPM(nn_model=NaiveUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device)
    
    if load_path is not None:
        ddpm.load_state_dict(torch.load(load_path))
        
    ddpm.to(device)
    
    dataset = iclevrDataset(mode='train', root='./iclevr')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = iclevrDataset(mode='test', root='./iclevr')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        
        # linear learning rate decay
        optim.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.8f}")
            optim.step()
        
        # generate and store it
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 * n_classes
            for i, cond in enumerate(test_dataloader):
                cond = cond.to(device)
                x_gen, x_gen_store = ddpm.sample(cond, batch_size, (3, 64, 64), device)
                xset = torch.cat([x_gen, x[:8]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                save_image(grid, os.path.join(output_dir, f'ddpm_sample_iclevr{ep}_{i}.png'))
                
        # store the model's parameters
        if save_model and ep % 5 == 0:
            torch.save(ddpm.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
            print('saved model at ' + os.path.join(save_dir, f"model_{ep}.pth"))
            
if __name__ == '__main__':
    train()