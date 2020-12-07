import numpy as np
import torch 
import os
import torchvision

from scipy.misc import imread
import argparse
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name',
                    type=str,
                    default='MineGAN_I2I_conditional_add_align_layers')
parser.add_argument('--guid_path',
                    type=str,
                    default='/DATA/data/Imagenet/crop_animal_faces_hdf/MineGAN_I2I_conditional_add_align_layers/test')
parser.add_argument('--save_path',
                    type=str,
                    default='/DATA/data/Imagenet/crop_animal_faces_hdf/MineGAN_I2I_conditional_add_align_layers/test_merge_image')
opts = parser.parse_args()

if not os.path.exists(os.path.join(opts.save_path, opts.experiment_name + '_output')):
    os.makedirs(os.path.join(opts.save_path, opts.experiment_name + '_output'))

cate = os.listdir(os.path.join(opts.guid_path, opts.experiment_name))
img_dirs = os.listdir(os.path.join(opts.guid_path, opts.experiment_name, cate[0]))

for img_dir in img_dirs:
    for img_index, cate_name in enumerate(cate):
        img = imread(os.path.join(opts.guid_path, opts.experiment_name, cate_name, img_dir))
        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
        img = img.unsqueeze(0); img = img.permute(0, 3, 1, 2)
        if img_index==0:
           img_input = imread(os.path.join(opts.guid_path,  opts.experiment_name + '_input_GT', img_dir)) 
           img_input = ((torch.from_numpy(img_input).float() / 255) - 0.5) * 2
           img_input = img_input.unsqueeze(0); img_input = img_input.permute(0, 3, 1, 2)
           fixed_t_x = img_input.detach().clone()
           fixed_t_x = torch.cat((img_input, fixed_t_x.detach().clone()), 0)
           img_dirs = [img_dir]
        else:
           fixed_t_x = torch.cat((fixed_t_x, img.detach().clone()), 0)
           img_dirs.append(img_dir)
        #if not os.path.exists(os.path.join(opts.save_path, img_dir)):
        #    os.makedirs(os.path.join(opts.save_path, img_dir))
    torchvision.utils.save_image(fixed_t_x, os.path.join(opts.save_path, opts.experiment_name + '_output', img_dir),
                                     nrow=int(fixed_t_x.shape[0] **0.5), normalize=True)





