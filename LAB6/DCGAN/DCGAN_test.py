import os
from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from DCGAN import Generator
from DCGAN_train import evaluate
from evaluator import evaluation_model
from dataloader import CLEVRDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # model
    parser.add_argument('--n_z', type=int, default=200)
    parser.add_argument('--num_conditions', type=int, default=24) # cannot modify
    parser.add_argument('--n_c', type=int, default=100)
    parser.add_argument('--n_ch_g', type=int, default=64)
    parser.add_argument('--n_ch_d', type=int, default=64)
    parser.add_argument('--img_sz', type=int, default=64)
    parser.add_argument('--add_bias', action='store_true', default=False)

    parser.add_argument('--bs', type=int, default=256)

    parser.add_argument('--cpt_path', type=str, default='./cpts/epoch250_iter17750_eval-acc0.5278.cpt')
    parser.add_argument('--output_dir', type=str, default='./DCGAN_output')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CLEVRDataset(img_dir='../iclevr/iclevr', json_file='../file/test.json', obj_file='../file/objects.json', transform=None, mode='test', img_sz=args.img_sz)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)

    generator = Generator(args).to(device)
    
    checkpoint = torch.load(args.cpt_path)
    generator.load_state_dict(checkpoint['g_model_state_dict'])
    
    eval_model = evaluation_model()

    avg_acc = 0
    for i in range(10):
        eval_acc, gen_images = evaluate(generator, loader, eval_model, args.n_z)
        avg_acc += eval_acc
        gen_images = 0.5 * gen_images + 0.5  # Denormalize
        save_image(gen_images, os.path.join(args.output_dir, f'DCGAN_{i}.png'), nrow=8)
        print(f'Accuracy: {eval_acc}')
    print(f'Average accuracy: {avg_acc / 10}')
