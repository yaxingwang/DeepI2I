#!/bin/bash
python train.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 1 --batch_size 4 --resume --resume_BigGAN   \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--N_target_cate 149 \
--G_ch 96 --D_ch 96 \
--test_every 2000000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--base_root result/animals \
--experiment_name  DeepI2I_animals  \
--data_root  ./data/animals 

##!/bin/bash
#python train.py \
#--dataset I128_hdf5 --parallel --shuffle  --num_workers 1 --batch_size 4 --training_scratch  \
#--num_G_accumulations 8 --num_D_accumulations 8 \
#--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
#--G_attn 64 --D_attn 64 \
#--G_nl inplace_relu --D_nl inplace_relu \
#--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
#--G_ortho 0.0 \
#--G_shared \
#--G_init ortho --D_init ortho \
#--hier --dim_z 120 --shared_dim 128 \
#--G_eval_mode \
#--N_target_cate 555 \
#--G_ch 96 --D_ch 96 \
#--test_every 2000000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#--use_multiepoch_sampler \
#--base_root result/NABirds \
#--experiment_name DeepI2I_NABirds  \
#--data_root  ./data/NABirds 




##!/bin/bash
#python train.py \
#--dataset I128_hdf5 --parallel --shuffle  --num_workers 1 --batch_size 4 --training_scratch  \
#--num_G_accumulations 8 --num_D_accumulations 8 \
#--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
#--G_attn 64 --D_attn 64 \
#--G_nl inplace_relu --D_nl inplace_relu \
#--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
#--G_ortho 0.0 \
#--G_shared \
#--G_init ortho --D_init ortho \
#--hier --dim_z 120 --shared_dim 128 \
#--G_eval_mode \
#--N_target_cate 256 \
#--G_ch 96 --D_ch 96 \
#--test_every 2000000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#--use_multiepoch_sampler \
#--base_root result/foods \
#--experiment_name  DeepI2I_UECFOOD256  \
#--data_root  ./data/foods 

