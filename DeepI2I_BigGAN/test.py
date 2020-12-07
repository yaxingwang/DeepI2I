""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
import pickle
import pdb

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # chaning the parameter for model from scratch
  if config['training_scratch']: 
      config['E1_fea_w'] = {4:1, 8:1, 16:1, 32:.1}
      config['D_fea_w'] = {4:0.1, 8:0.1, 16:0.1, 32:.01}

  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  # Minor
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)
  E1 = model.Encoder(**config).to(device)
  A1 = model.Alignment(**config).to(device)
  
   # If using EMA, prepare it
  if config['ema']:#  here it is True 
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:#  here it is False 
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:#  here it is False
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D, E1, A1)
  print(G)
  print(D)
  print(E1)
  print(A1)
  print('Number of params in G: {} D: {} E1: {} A1: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D,E1, A1]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G, D, E1, A1, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None,  model_ite=config['model_ite'])#  :I add load_optim=Fasle

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD) # BigGAN use it
    #GD =  nn.DistributedDataParallel(GD)# Yaxing update it
    
    if config['cross_replica']:#  here it is False
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr'], 'target_domain':None})

  loaders_t = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr'], 'target_domain':None})

  if config['model_ite'] > 0:
      config['test_root'] = config['test_root'] + '/'+ str(config['model_ite'])

  with open('class_to_index/%s/I128_imgs.pickle'%experiment_name, 'rb') as handle:
    class_to_index = pickle.load(handle)
    index_to_class={class_to_index[i]:i for i in class_to_index.keys()}

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], N_target_cate = config['N_target_cate'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN': #  : here it is  GAN 
    train = train_fns.GAN_training_function(G, D, E1, A1, GD, z_, y_, 
                                            ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()

  # target data
  t_data = iter(loaders_t[0])# : target domain
  fixed_t_x, fixed_t_y = None, None

  # Train for specified number of epochs, although we mostly track G iterations.
  print('Switchin G to eval mode...')
  E1.eval()
  A1.eval()
  G.eval()
  if config['ema']:
    G_ema.eval()


  for index_epoch_num in range(0, 1):    
    t_batch = next(t_data)#target domain
    t_x, t_y = t_batch
    if len(t_x) != (config['num_D_accumulations'] * config['batch_size']):
        t_data = iter(loaders_t[0])#since it will stop when it read all loop, we need reset
        t_batch = next(t_data)#target domain
        t_x, t_y = t_batch

    fixed_t_x, fixed_t_y = t_x[:G_batch_size].detach().clone(), t_y[:G_batch_size].detach().clone()
    fixed_t_x, fixed_t_y = fixed_t_x.to(device), fixed_t_y.to(device)

    train_fns.sample_all_cate(G, D, E1, A1, G_ema, z_, y_, fixed_z, fixed_y, 
                              state_dict, config, experiment_name, fixed_t_x, fixed_t_y, None, config['N_target_cate'], index_to_class=index_to_class,index_epoch_num=index_epoch_num)
    


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()
