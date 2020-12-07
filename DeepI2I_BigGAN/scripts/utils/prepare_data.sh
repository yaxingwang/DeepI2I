#!/bin/bash
#python make_hdf5.py --dataset I128 --batch_size 256 --data_root /DATA/data/catdog_drit/cat 
python make_hdf5.py --dataset I128 --batch_size 256 --data_root /DATA/data/catdog_drit/dog 
#python calculate_inception_moments.py --dataset I128_hdf5 --data_root /media/yaxing/Elements/IIAI_raid/Imagenet/single_cate

