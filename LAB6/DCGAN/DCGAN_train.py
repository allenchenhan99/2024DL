import os
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from evaluator import evaluation_model
from dataloader import CLEVRDataset
from DCGAN import Generator, Discriminator, weights_init

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 8

def sample_z(bs, n_z, mode='normal'):
    if mode == 'normal':
        return torch.normal(torch.zeros((bs, n_z)), torch.ones((bs, n_z)))
    elif mode == 'uniform':
        return torch.randn(bs, n_z)
    else:
        raise NotImplementedError()

def evaluate(g_model, loader, eval_model, n_z):
    g_model.eval()
    avg_acc = 0
    gen_images = None
    with torch.no_grad():
        for _, (_, conds) in enumerate(loader):
            conds = conds.to(device)
            z = sample_z(conds.shape[0], n_z).to(device)
            fake_images = g_model(z, conds)
            gen_images = fake_images if gen_images is None else torch.vstack((gen_images, fake_images))
            acc = eval_model.eval(fake_images, conds)
            avg_acc += acc * conds.shape[0]
    avg_acc /= len(loader.dataset)
    return avg_acc, gen_images

def train(g_model, d_model, optimizer_g, optimizer_d, criterion, num_epochs, train_loader, test_loader, n_z, eval_interval, cpt_dir, result_dir, start_epoch=0):
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    eval_model = evaluation_model()
    best_acc = 0
    batch_done = start_epoch * len(train_loader)  # compute the number of completed batch
    pbar_epoch = tqdm(range(start_epoch, num_epochs))
    
    for epoch in pbar_epoch:
        g_model.train()
        d_model.train()
        losses_g = 0
        losses_d = 0
        pbar_batch = tqdm(train_loader)
        
        for batch_idx, (images, conds) in enumerate(pbar_batch):
            images = images.to(device)
            conds = conds.to(device)
            bs = images.shape[0]

            real_label_g = torch.ones(bs).to(device)  # make sure the shape consistent
            fake_label_g = torch.zeros(bs).to(device)

            real_label_d = torch.ones(bs).to(device)
            fake_label_d = torch.zeros(bs).to(device)

            # Discriminator
            optimizer_d.zero_grad()
            outputs = d_model(images, conds).view(-1)  # make sure the shape consistent
            loss_real = criterion(outputs, real_label_d)
            d_x = outputs.mean().item()
            z = sample_z(bs, n_z).to(device)
            fake_images = g_model(z, conds)
            outputs = d_model(fake_images.detach(), conds).view(-1)  # make sure the shape consistent
            loss_fake = criterion(outputs, fake_label_d)
            d_g_z1 = outputs.mean().item()
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Generator
            optimizer_g.zero_grad()
            z = sample_z(bs, n_z).to(device)
            fake_images = g_model(z, conds)
            outputs = d_model(fake_images, conds).view(-1)  # make sure the shape consistent
            loss_g = criterion(outputs, real_label_g)
            d_g_z2 = outputs.mean().item()
            loss_g.backward()
            optimizer_g.step()

            pbar_batch.set_description('[{}/{}][{}/{}][LossG={:.4f}][LossD={:.4f}][D(x)={:.4f}][D(G(z))={:.4f}/{:.4f}]'
                .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss_g.item(), loss_d.item(), d_x, d_g_z1, d_g_z2))

            losses_g += loss_g.item()
            losses_d += loss_d.item()
            batch_done += 1

            if batch_done % eval_interval == 0:
                eval_acc, gen_images = evaluate(g_model, test_loader, eval_model, n_z)
                # print(f'range=({gen_images.min()}, {gen_images.max()})', file=open('gan_value_range.txt', 'w'))
                gen_images = 0.5 * gen_images + 0.5
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    torch.save(
                        {
                            'epoch': epoch,
                            'g_model_state_dict': g_model.state_dict(),
                            'd_model_state_dict': d_model.state_dict(),
                            'optimizer_g_state_dict': optimizer_g.state_dict(),
                            'optimizer_d_state_dict': optimizer_d.state_dict(),
                            'best_acc': best_acc
                        },
                        os.path.join(cpt_dir, f'epoch{epoch + 1}_iter{batch_done}_eval-acc{eval_acc:.4f}.cpt')
                    )
                save_image(gen_images, os.path.join(result_dir, f'epoch{epoch + 1}_iter{batch_done}.png'), nrow=8)
                save_image(gen_images, 'gan_current.png', nrow=8)
                g_model.train()
                d_model.train()

        avg_loss_g = losses_g / len(train_loader)
        avg_loss_d = losses_d / len(train_loader)
        eval_acc, gen_images = evaluate(g_model, test_loader, eval_model, n_z)
        gen_images = 0.5 * gen_images + 0.5
        pbar_epoch.set_description(
            '[{}/{}][AvgLossG={:.4f}][AvgLossD={:.4f}][EvalAcc={:.4f}]'
            .format(epoch + 1, num_epochs, avg_loss_g, avg_loss_d, eval_acc)
        )
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(
                {
                    'epoch': epoch,
                    'g_model_state_dict': g_model.state_dict(),
                    'd_model_state_dict': d_model.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'best_acc': best_acc
                },
                os.path.join(cpt_dir, f'epoch{epoch + 1}_last_eval-acc{eval_acc:.4f}.cpt')
            )
        save_image(gen_images, os.path.join(result_dir, f'epoch{epoch + 1}_last.png'), nrow=8)
        save_image(gen_images, 'gan_current.png', nrow=8)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    # model parameters
    parser.add_argument('--n_z', type=int, default=150)
    parser.add_argument('--num_conditions', type=int, default=24)
    parser.add_argument('--n_c', type=int, default=100)
    parser.add_argument('--n_ch_g', type=int, default=64)
    parser.add_argument('--n_ch_d', type=int, default=64)
    parser.add_argument('--img_sz', type=int, default=64)
    parser.add_argument('--add_bias', action='store_true', default=False)

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim_d', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--bs', type=int, default=256)

    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--cpt_dir', type=str, default='DCGAN_cpts')
    parser.add_argument('--result_dir', type=str, default='DCGAN_results')
    parser.add_argument('--resume', type=str, default='DCGAN_cpts/epoch16_iter1100_eval-acc0.2222.cpt')  # for loading the checkpoint
    # parser.add_argument('--resume', type=str)  # for loading the checkpoint

    args = parser.parse_args()

    generator = Generator(args).to(device)
    generator.apply(weights_init)

    discriminator = Discriminator(args).to(device)
    discriminator.apply(weights_init)

    optimizer_g = torch.optim.Adam(generator.parameters(), args.lr, betas=(args.beta1, args.beta2))
    if args.optim_d == 'adam':
        optimizer_d = torch.optim.Adam(discriminator.parameters(), args.lr, betas=(args.beta1, args.beta2))
    elif args.optim_d == 'sgd':
        optimizer_d = torch.optim.SGD(discriminator.parameters(), args.lr, momentum=0.9)
    else:
        raise NotImplementedError()

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        generator.load_state_dict(checkpoint['g_model_state_dict'])
        discriminator.load_state_dict(checkpoint['d_model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    train_trans = transforms.Compose([
        transforms.Resize((args.img_sz, args.img_sz)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CLEVRDataset(img_dir='../iclevr/iclevr', json_file='../file/train.json', obj_file='../file/objects.json', transform=train_trans, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers)

    test_dataset = CLEVRDataset(img_dir='../iclevr/iclevr', json_file='../file/test.json', obj_file='../file/objects.json', transform=train_trans, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=num_workers)

    criterion = nn.BCELoss()

    train(
        g_model=generator,
        d_model=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        criterion=criterion,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        n_z=args.n_z,
        eval_interval=args.eval_interval,
        cpt_dir=args.cpt_dir,
        result_dir=args.result_dir,
        start_epoch=start_epoch
    )