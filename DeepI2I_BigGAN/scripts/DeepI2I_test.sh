#!/bin/bash
for model_ite in $(seq 0 10000 1)   
do
    python test.py \
    --dataset I128_hdf5 --parallel --shuffle  --num_workers 1 --batch_size 32 --resume  \
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
    --model_ite $model_ite \
    --N_target_cate 149 \
    --G_ch 96 --D_ch 96 \
    --test_every 2000000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2020 \
    --use_multiepoch_sampler \
    --base_root result/animals \
    --experiment_name  DeepI2I_animals  \
    --data_root  ./data/animals 

    experiment_name=DeepI2I_animals
    python merge_image.py --experiment_name  $experiment_name \
    --guid_path  result/animals/test/$model_ite \
    --save_path  result/animals/test/$model_ite

done


##!/bin/bash
#for model_ite in $(seq 30000 1000 30001)
#do
#    python test.py \
#    --dataset I128_hdf5 --parallel --shuffle  --num_workers 1 --batch_size 64 --resume  \
#    --num_G_accumulations 8 --num_D_accumulations 8 \
#    --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
#    --G_attn 64 --D_attn 64 \
#    --G_nl inplace_relu --D_nl inplace_relu \
#    --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
#    --G_ortho 0.0 \
#    --G_shared \
#    --G_init ortho --D_init ortho \
#    --hier --dim_z 120 --shared_dim 128 \
#    --G_eval_mode \
#    --model_ite $model_ite \
#    --N_target_cate 256 \
#    --G_ch 96 --D_ch 96 \
#    --test_every 2000000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2020 \
#    --use_multiepoch_sampler \
#    --base_root result/foods \
#    --experiment_name  DeepI2I_UECFOOD256  \
#    --data_root  ./data/foods
#
#    experiment_name=DeepI2I_UECFOOD256
#    python merge_image.py --experiment_name  $experiment_name \
#    --guid_path  result/foods/test/$model_ite \
#    --save_path  result/foods/test/$model_ite
#
#done




##!/bin/bash
#for model_ite in $(seq 30000 1000 30001)
#do
#    python test.py \
#    --dataset I128_hdf5 --parallel --shuffle  --num_workers 1 --batch_size 64 --resume  \
#    --num_G_accumulations 8 --num_D_accumulations 8 \
#    --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
#    --G_attn 64 --D_attn 64 \
#    --G_nl inplace_relu --D_nl inplace_relu \
#    --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
#    --G_ortho 0.0 \
#    --G_shared \
#    --G_init ortho --D_init ortho \
#    --hier --dim_z 120 --shared_dim 128 \
#    --G_eval_mode \
#    --model_ite $model_ite \
#    --N_target_cate 555 \
#    --G_ch 96 --D_ch 96 \
#    --test_every 2000000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2020 \
#    --use_multiepoch_sampler \
#    --base_root result/NABirds \
#    --experiment_name  DeepI2I_NABirds  \
#    --data_root  ./data/NABirds
#
#    experiment_name=DeepI2I_NABirds
#    python merge_image.py --experiment_name  $experiment_name \
#    --guid_path  result/NABirds/test/$model_ite \
#    --save_path  result/NABirds/test/$model_ite
#
#done
#
