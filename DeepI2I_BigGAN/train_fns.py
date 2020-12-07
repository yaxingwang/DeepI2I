''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses
import pdb


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, E1, A1, GD, z_, y_, ema, state_dict, config):#  

  def train(x, y, t_x, t_y, stage=1, training_scratch=False):
    G.optim.zero_grad()
    D.optim.zero_grad()
    E1.optim.zero_grad()#   
    A1.optim.zero_grad()#   
    x = torch.split(x, config['batch_size'])# How many chunks to split x and y into?
    y = torch.split(y, config['batch_size'])# How many chunks to split x and y into?
    t_x = torch.split(t_x, config['batch_size'])# How many chunks to split x and y into?
    t_y = torch.split(t_y, config['batch_size'])# How many chunks to split x and y into?
    D_fea_w = config['D_fea_w']
    #  add this
    G_para_update = ['shared', 'linear', 'bn', 'output_layer', 'blocks.3.1.gamma', 'blocks.3.1.theta', 'blocks.3.1.phi', 'blocks.3.1.g', 'blocks.3.1.o']
    counter = 0

    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:#  here it is True 
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      utils.toggle_grad(E1, False) #  
      utils.toggle_grad(A1, False) #  
      
    for step_index in range(config['num_D_steps']):


      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], t_x=t_x[counter], t_y=t_y[counter], train_G=False, 
                            split_D=config['split_D'])
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:#  here it is 0.0 
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      T_F = True if stage==2 else False 
      utils.toggle_grad(G, T_F)#  
      utils.toggle_grad(A1,True) #  
      if training_scratch: 
        utils.toggle_grad(E1, True) #  
      else:
        utils.toggle_grad(E1, False) #  
      

      
    # Zero G's gradients by default before training G, for safety
    A1.optim.zero_grad()#  
    if stage==2: 
        G.optim.zero_grad()
    if training_scratch: 
        E1.optim.zero_grad()#  
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']): #  here it is 1 
      z_.sample_()
      y_.sample_()
      #  : set gy and dy is equal 0, since we donot know label 
      D_fake, M_regu= GD(z_, y_, t_x=t_x[accumulation_index], t_y=t_y[accumulation_index],  train_G=True, split_D=config['split_D'],  M_regu=True,train_E1=True, train_A1=True)

      M_E1_loss = losses.generator_loss(D_fake, M_regu, D_fea_w=D_fea_w) / float(config['num_G_accumulations'])
      #pdb.set_trace()
      M_E1_loss.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:#  here it is 0.0 
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    A1.optim.step()
    if stage==2: 
        G.optim.step()
    if training_scratch: 
        E1.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    #out = {'G_loss': float(G_loss.item()), 
    out = {'G_loss': float(M_E1_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train
  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, E1, A1, G_ema, z_, y_, fixed_z, fixed_y, 
                   state_dict, config, experiment_name, fixed_t_x, fixed_t_y, fixed_x_v2):
  utils.save_weights(G, D, E1, A1, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  if not (state_dict['itr'] % config['save_every']):
      utils.save_weights(G, D, E1, A1, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None, copy=True)
 # # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      E1_L_feat =  nn.parallel.data_parallel(E1, (fixed_t_x, fixed_t_y,  True))

      E1_L_feat =  nn.parallel.data_parallel(A1, (E1_L_feat, None, None))

      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y), E1_L_feat))
    else:
      E1_L_feat =  E1(fixed_t_x, fixed_t_y,  True)
      E1_L_feat =  A1(E1_L_feat, None, None)
      fixed_Gz = which_G(fixed_z, fixed_t_y, E1_L_feat)
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)

  # source real image
  source_real_name = image_filename.split('fixed')[0] + 'soure_real.jpg'
  torchvision.utils.save_image(fixed_x_v2.float().cpu(), source_real_name,
                             nrow=int(fixed_x_v2.shape[0] **0.5), normalize=True)

  # target real image
  target_real_name = image_filename.split('fixed')[0] + 'target_real.jpg'
  torchvision.utils.save_image(fixed_t_x.float().cpu(), target_real_name,
                             nrow=int(fixed_t_x.shape[0] **0.5), normalize=True)

  # For now, every time we save, also save sample sheets

  utils.sample_sheet(which_G,E1_L_feat,
                     classes_per_sheet=E1_L_feat[32].shape[0] if (E1_L_feat[32].shape[0]<utils.classes_per_sheet_dict[config['dataset']]) else utils.classes_per_sheet_dict[config['dataset']],# original one :classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']] for large batch. Yaxing's for small batch
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)



def sample_all_cate(G, D, E1, A1, G_ema, z_, y_, fixed_z, fixed_y, 
                   state_dict, config, experiment_name, fixed_t_x, fixed_t_y, fixed_x_v2, num_target_cate,index_to_class,index_epoch_num):

  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:

      E1_L_feat =  nn.parallel.data_parallel(E1, (fixed_t_x, fixed_t_y,  True))

      E1_L_feat =  nn.parallel.data_parallel(A1, (E1_L_feat, None, None))
    else:
      E1_L_feat =  E1(fixed_t_x, fixed_t_y,  True)
      E1_L_feat =  A1(E1_L_feat, None, None)

  if not os.path.isdir('%s/%s' % (config['test_root'], experiment_name)):
    os.makedirs('%s/%s' % (config['test_root'], experiment_name))
  # target real image

  if not os.path.isdir('%s/%s' % (config['test_root'], experiment_name + '_input_GT')):
    os.mkdir('%s/%s' % (config['test_root'], experiment_name + '_input_GT'))
  
  for index_ in range(len(fixed_t_x)):
      target_real_name = '%s/%s/%d.jpg' % (config['test_root'], 
                                                  experiment_name + '_input_GT',
                                                  index_epoch_num*len(fixed_t_x) + index_)
      torchvision.utils.save_image(fixed_t_x[index_].float().cpu(), target_real_name,
                             nrow=1, normalize=True)

  target_real_name = '%s/%s/%s.jpg' % (config['test_root'], 
                                                  experiment_name + '_input_GT',
                                                  str(index_epoch_num*len(fixed_t_x)) + '_set')
  torchvision.utils.save_image(fixed_t_x.float().cpu(), target_real_name,
                             nrow=int(fixed_t_x.shape[0] **0.5), normalize=True)

  # generating samplies for all categories 
  single_sampe =E1_L_feat 
  utils.sample_sheet_all_cate(which_G, single_sampe,
                     classes_per_sheet=single_sampe[32].shape[0] if (single_sampe[32].shape[0]<utils.classes_per_sheet_dict[config['dataset']]) else utils.classes_per_sheet_dict[config['dataset']],# original one :classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']] for large batch. Yaxing's for small batch
                     num_classes=config['n_classes'],
                     samples_per_class=1, parallel=config['parallel'],
                     samples_root=config['test_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_,
                     num_target_cate=num_target_cate,
                     index_to_class=index_to_class,
                     index_epoch_num=index_epoch_num)
