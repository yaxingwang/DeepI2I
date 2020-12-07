# Overview 
- [Dependences](#dependences)
- [Installation](#installtion)
- [Instructions](#instructions)
- [Results](#results)
- [References](#references)
- [Contact](#contact)
# Dependences 
- Python2.7, NumPy, SciPy, NVIDIA GPU
- **Pytorch:** pytorch is more 1.0
- **Dataset:** [animals](https://github.com/NVlabs/FUNIT), [NABirds](https://dl.allaboutbirds.org/nabirds) and [UECFOOD-256](http://foodcam.mobi/dataset256.html) 

# Installation 
- Install pytorch
# Instructions

- `git clone git@github.com:yaxingwang/MineGAN.git` to get `MineGA`

- Pretrained model: downloading the pretrained model from [Biggan](https://github.com/ajbrock/BigGAN-PyTorch). Note using `G_ema.pth` to replace `G.pth`, since we dones't use `ema`. The pretrained model is moved into `BigGAN_weights/` 


- Preparing data: leveraging  `sh scripts/utils/prepare_data.py`, and put it into `data/your_data/data`. Please check [Biggan](https://github.com/ajbrock/BigGAN-PyTorch) to learn how to generate the data 

I have already created a [example](https://drive.google.com/drive/folders/1Wvmz_SHlJekHjuC4UJCncxdcJsYlwcCb?usp=sharing). Downloading the three items and put them into `data/animals`. Also I upload the compressed NABirds and UECFOOD-256, which is only be used for our project.  

- Traing: ```sh scripts/DeepI2I.sh```

The corresponding model and generated images are saved in   `result/animals` where four items are automatically generated: 'logs', 'samples', 'test' and 'weights'.  


- Testing: ```sh scripts/DeepI2I_test.sh```

Note if you use new name (e.g., '--experiment DeepI2I_animalv2' in 'scripts/DeepI2I.sh'), you should also use it in  'scripts/DeepI2I_test.sh', and rename the  fold ( 'class_to_index/DeepI2I_animals') to the new one ( 'class_to_index/DeepI2I_animalv2') 

Downloading  our [pre-trained model](https://drive.google.com/drive/folders/19pSSiNDmebtm17ymw3tYe5V5G9wI6RHR?usp=sharing) on animals, and put it into 'result/animals/weights/DeepI2I_animals/0'. Also the pre-trained model for [birds](https://drive.google.com/drive/folders/1gZpkFzLp9w8X1PsTiqPrPJWll5DgX2XP?usp=sharing) and [foods](updating)


```

If you use the provided data and code, please cite the following papers:
 
```

@article{yu2020deepi2i,
  title={DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs},
  author={Wang, Yaxing, Yu, Lu and van de Weijer, Joost},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

```
