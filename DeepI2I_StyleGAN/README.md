# DeepI2I for StyleGAN

The code is heavily based on the [StyleGAN-pytorch](https://github.com/rosinality/style-based-gan-pytorch).

See `stylegan` how to creat data and download pre-trained StyleGAN.

### Dependences 
- Python3.7, NumPy, SciPy, NVIDIA GPU
- **Pytorch:**  Pytorch is more 1.2 (pytorch14 doesn't work)

### Preprocess datasets
```
python prepare_data.py --out dataset/DATASET_lmdb --n_worker 8 dataset/DATASET
```

This will convert images to jpeg and pre-resizes it (For example, 8/16/32/64/128/256/512/1024).

### Download pre-traind GAN models
```
# Download from https://drive.google.com/file/d/1Q04ojQgKdDdFEnnGJSavcIr4dDTDZEP-/view?usp=sharing
# Save model in ./checkpoint/StyleGAN_pretrained directory
```


### Run experiments
```
python train.py  data/cat --path_y data/dog --init_size 256 --max_size 256 --mixing  --ckpt checkpoint/StyleGAN_pretrained/stylegan-256px-new.model  


```

Here the path (data/cat) contains both data.mdb and lock.mdb.  


If you use the provided data and code, please cite the following papers:
 
```

@article{wang2020deepi2i,
  title={DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs},
  author={Wang, Yaxing and Yu, Lu and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2011.05867},
  year={2020}
}

```
