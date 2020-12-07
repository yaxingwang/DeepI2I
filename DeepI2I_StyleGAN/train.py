import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
torch.backends.cudnn.benchmark = True
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, Alignment 



import pdb


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, dataset, generator, discriminator, encoder, align, dataset_y):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    loader_y = sample_data(
        dataset_y, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)
    data_loader_y = iter(loader_y)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
    # E1
    adjust_lr(e_optimizer, args.lr.get(resolution, 0.001))
    # A1
    adjust_lr(a_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    requires_grad(generator, False)
    #E1
    requires_grad(encoder, False)
    #A1
    requires_grad(align, False)

    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False
    stage_change = 500
    value_column = 8
    for i in pbar:
        discriminator.zero_grad()

        #alpha = min(1, 1 / args.phase * (used_sample + 1))
        # the pre-trained model is obtained when alpha is 1
        alpha = 1

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)
            loader_y = sample_data(
                dataset_y, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader_y = iter(loader_y)

            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'encoder': encoder.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'e_optimizer': e_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)
            real_image_y = next(data_loader_y)
            if i==0:
                fix_real_image=[]
                fix_real_image_y=[]
                for _ in range(10): # (10, 5) is to visualize the generated images, which is correspoinding the following codes (10, 5)
                    fix_real_image.append(real_image[:value_column].cuda())
                    fix_real_image_y.append(real_image_y[:value_column].cuda())

                    real_image = next(data_loader)
                    real_image_y = next(data_loader_y)


        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

            data_loader_y = iter(loader_y)
            real_image_y = next(data_loader_y)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        real_image_y = real_image_y.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9: # mixing is True
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)
        # E1
        _, L_feat = encoder(real_image_y, step=step, alpha=alpha, E1_output_feat=True, RESOLUTION=args.RESOLUTION)
        # A1
        L_feat = align(L_feat)

        fake_image = generator(gen_in1, step=step, alpha=alpha, E1_output_feat=True, L_feat=L_feat, RESOLUTION=args.RESOLUTION, E1_fea_w=args.E1_fea_w)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            align.zero_grad()
            requires_grad(align, True)
            # E1
            if i > stage_change: 
                #encoder.zero_grad()
                #requires_grad(encoder, True)
                generator.zero_grad()
                requires_grad(generator, True)
            else:
                generator.zero_grad()
                requires_grad(generator, False)

            requires_grad(discriminator, False)

            # E1
            _, L_feat = encoder(real_image_y, step=step, alpha=alpha, E1_output_feat=True, RESOLUTION=args.RESOLUTION)
            # A1
            L_feat = align(L_feat)
            fake_image = generator(gen_in2, step=step, alpha=alpha, E1_output_feat=True, L_feat=L_feat, RESOLUTION=args.RESOLUTION, E1_fea_w=args.E1_fea_w)

            predict, L_out_feat = discriminator(fake_image, step=step, alpha=alpha, E1_output_feat=True, RESOLUTION=args.RESOLUTION)
            _, L_in_feat = discriminator(real_image_y, step=step, alpha=alpha, E1_output_feat=True, RESOLUTION=args.RESOLUTION)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()
                # reconstruction loss
                loss_M = 0.
                for keys in args.RESOLUTION: 
                        loss_M += args.D_fea_w[keys] * torch.mean(F.mse_loss(L_in_feat[keys], L_out_feat[keys])) 
                loss += loss_M

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()
                # reconstruction loss
                loss_M = 0.
                for keys in args.RESOLUTION: 
                    loss_M += args.D_fea_w[keys] * torch.mean(F.mse_loss(L_in_feat[keys], L_out_feat[keys])) 
                loss += loss_M

            if i%10 == 0:
                gen_loss_val = loss.item()

            loss.backward()
            a_optimizer.step()
            requires_grad(align, False)
            # E1
            if i > stage_change: 
                #e_optimizer.step()
                #requires_grad(encoder, False)
                g_optimizer.step()
                accumulate(g_running, generator.module, 0)

            requires_grad(generator, False)
            requires_grad(encoder, False)
            requires_grad(discriminator, True)

        if i  % 100 == 0:
            images = []
            images_source = []
            images_y = []

            encoder.eval()
            align.eval()
            gen_i, gen_j = args.gen_sample.get(resolution, (10, value_column))


            with torch.no_grad():
                for i_ in range(gen_i):
                    
                    _, L_feat_fixed = encoder(fix_real_image_y[i_], step=step, alpha=alpha, E1_output_feat=True, RESOLUTION=args.RESOLUTION)
                    L_feat_fixed = align(L_feat_fixed)
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(), step=step, alpha=alpha, E1_output_feat=True, L_feat=L_feat_fixed, RESOLUTION=args.RESOLUTION, E1_fea_w=args.E1_fea_w
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )
            if i==0: # source and target real images
                utils.save_image(
                    torch.cat([img.data.cpu() for img in fix_real_image_y], 0),
                    f'sample/y.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    torch.cat([img.data.cpu() for img in fix_real_image], 0),
                    f'sample/target.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                )
            align.train()
        if (i + 1) % 5000 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'encoder': encoder.module.state_dict(),
                    'align': align.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'e_optimizer': e_optimizer.state_dict(),
                    'a_optimizer': a_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_step-{str(i + 1).zfill(6)}.model',
            )

           # torch.save(
           #     g_running.state_dict(), f'checkpoint/G_running_{str(i + 1).zfill(6)}.model'
           # )
           # torch.save(
           #     generator.state_dict(), f'checkpoint/G_{str(i + 1).zfill(6)}.model'
           # )
           # torch.save(
           #     g_optimizer.state_dict(), f'checkpoint/G_optim_{str(i + 1).zfill(6)}.model'
           # )

           # torch.save(
           #     discriminator.state_dict(), f'checkpoint/D_{str(i + 1).zfill(6)}.model'
           # )
           # torch.save(
           #     d_optimizer.state_dict(), f'checkpoint/D_optim_{str(i + 1).zfill(6)}.model'
           # )

           # torch.save(
           #     encoder.state_dict(), f'checkpoint/E_{str(i + 1).zfill(6)}.model'
           # )
           # torch.save(
           #     e_optimizer.state_dict(), f'checkpoint/E_optim_{str(i + 1).zfill(6)}.model'
           # )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path',  type=str, help='path of specified dataset')
    parser.add_argument('--path_y', default='data/dog', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )


    parser.add_argument('--RESOLUTION', default=[64,32,16,8,4], type=list, help='the selected features')
    parser.add_argument('--E1_fea_w', default={4:.1, 8:.1, 16:.1, 32:.1, 64:.0}, type=dict, help='weights of each encoder feautures which is used in generator')
    parser.add_argument('--D_fea_w', default={4:.1, 8:.1, 16:.1, 32:.1, 64:0.}, type=dict, help='We compute the distance between real image and fake image, to keep the structure information')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    # E1
    encoder = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()

    # A1
    align = nn.DataParallel(
        Alignment(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()

    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    # E1
    e_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.0, 0.99))
    # A1
    a_optimizer = optim.Adam(align.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)#Yaxing

    # Big probelm
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        # E1
        encoder.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        # E1
        e_optimizer.load_state_dict(ckpt['d_optimizer'])
        # A1
        #a_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)
    dataset_y = MultiResolutionDataset(args.path_y, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    #args.batch_default = 32
    args.batch_default = 32#yaxing

    train(args, dataset, generator, discriminator, encoder, align, dataset_y)
