import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from Unet import ContextUnet, NaiveUnet
from DDPM import DDPM
from dataset import iclevrDataset
from evaluator import evaluation_model

if __name__ == '__main__':
    parser = ArgumentParser()

    # Parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_T', type=int, default=1500)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_classes', type=int, default=24)
    parser.add_argument('--n_feat', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--root', type=str, default='./iclevr')
    parser.add_argument('--cpts', type=str, default='./cpts/diffusion_outputs10_lemb_tembV2_b32_NaiveUnetV6/model_90.pth')
    parser.add_argument('--ws_test', nargs='+', type=float, default=[0.0, 0.5, 2.0])
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--new_test_file', type=str, default='new_test.json')

    parser.add_argument('--result_dir', type=str, default='./result')

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    test_dataset = iclevrDataset(mode='test', root=args.root, test_file=args.test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    new_test_dataset = iclevrDataset(mode='test', root=args.root, test_file=args.new_test_file)
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=args.batch_size, shuffle=False)

    ddpm = DDPM(nn_model=NaiveUnet(in_channels=3, n_feat=args.n_feat, n_classes=args.n_classes), betas=(1e-4, 0.02), n_T=args.n_T, device=args.device)
    ddpm.load_state_dict(torch.load(args.cpts))
    ddpm.to(args.device)
    judger = evaluation_model()

    num_denoise_steps = 11

    ddpm.eval()
    with torch.no_grad():
        for i, cond in enumerate(test_dataloader):
            cond = cond.to(args.device)
            x_gen, x_gen_store = ddpm.sample(cond, args.batch_size, (3, 64, 64), args.device)
            acc = judger.eval(x_gen, cond)
            print(f'The accuracy of data in {args.test_file} is {acc:.4f}')
            
            # Save the denoising process images
            for j in range(cond.shape[0]):  # Iterate over the batch
                step_indices = torch.linspace(0, len(x_gen_store)-1, steps=num_denoise_steps, dtype=torch.long)
                imgs = [torch.tensor(x_gen_store[idx][j]).unsqueeze(0) for idx in step_indices]
                imgs = torch.cat(imgs, dim=0)  # Create a tensor of shape (num_denoise_steps, 3, 64, 64)
                grid = make_grid(imgs, nrow=num_denoise_steps, normalize=True, value_range=(-1, 1))  # Set nrow to num_denoise_steps to create a single row
                save_image(grid, os.path.join(args.result_dir, f'ddpm_denoise_step_{i}_{j}.png'))
        
        for i, cond in enumerate(new_test_dataloader):
            cond = cond.to(args.device)
            x_gen, x_gen_store = ddpm.sample(cond, args.batch_size, (3, 64, 64), args.device)
            acc = judger.eval(x_gen, cond)
            print(f'The accuracy of data in {args.new_test_file} is {acc:.4f}')
            
            # Save the denoising process images
            for j in range(cond.shape[0]):  # Iterate over the batch
                step_indices = torch.linspace(0, len(x_gen_store)-1, steps=num_denoise_steps, dtype=torch.long)
                imgs = [torch.tensor(x_gen_store[idx][j]).unsqueeze(0) for idx in step_indices]
                imgs = torch.cat(imgs, dim=0)  # Create a tensor of shape (num_denoise_steps, 3, 64, 64)
                grid = make_grid(imgs, nrow=num_denoise_steps, normalize=True, value_range=(-1, 1))  # Set nrow to num_denoise_steps to create a single row
                save_image(grid, os.path.join(args.result_dir, f'ddpm_denoise_step_new_{i}_{j}.png'))
