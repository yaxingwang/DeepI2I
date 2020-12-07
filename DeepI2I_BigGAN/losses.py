import torch
import torch.nn.functional as F
import pdb

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake, M_regu=None, D_fea_w={4:10, 8:5, 16:2, 32:0.1}):
  loss = torch.mean(F.softplus(-dis_fake))
  loss_M = 0.
  if M_regu is not None:
       for keys in M_regu[-1].keys(): 
           loss_M += D_fea_w[keys] * torch.mean(F.mse_loss(M_regu[-1][keys][0], M_regu[-1][keys][1])) 
  loss +=  loss_M  
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake, M_regu=None, D_fea_w={4:10, 8:5, 16:2, 32:0.1}):
  loss = -torch.mean(dis_fake)
  loss_M = 0.
  if M_regu is not None:
       for keys in M_regu[-1].keys(): 
           loss_M += D_fea_w[keys] * torch.mean(F.mse_loss(M_regu[-1][keys][0], M_regu[-1][keys][1])) 
  loss +=  loss_M  
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
#generator_loss = loss_dcgan_gen
#discriminator_loss = loss_dcgan_dis
