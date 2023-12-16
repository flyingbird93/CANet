# CANet
 the Cross Attention Network (CANet), a new framework that formulates image enhancement as a parallel learning process based on the image and LUT features

## Downloads
### [Paper]([https://papers.bmvc2023.org/0348.pdf]), Datasets([[GoogleDrive](https://drive.google.com/drive/folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a?usp=sharing)],[[onedrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/16901447r_connect_polyu_hk/EqNGuQUKZe9Cv3fPG08OmGEBbHMUXey2aU03E21dFZwJyg?e=QNCMMZ)],[[baiduyun](https://pan.baidu.com/s/1CsQRFsEPZCSjkT3Z1X_B1w):5fyk])
Here I only provided the FiveK dataset resized into 480p resolution (including 8-bit sRGB and 8-bit sRGB targets). [FiveK](https://data.csail.mit.edu/graphics/fivek/) dataset.

## Abstract
Image enhancement aims to improve the quality of images by adjusting their color and is widely used in professional digital photography. Deep learning-based 3 Dimensional LookUp-Table (3D LUT) of RGB color transformation has achieved promising performance in terms of speed and precision. However, the focus has mainly been on building an adaptive enhancer by only learning the global color adjusting weights from the image, which ignores the significant relationship between the intrinsic semantic information of the image and LUT that is relevant to photographers. In this paper, we propose the Cross Attention Network (CANet), a new framework that formulates image enhancement as a parallel learning process based on the image and LUT features. To better learn the adjustment weights for both global color and intrinsic semantics, we propose a cross attention architecture that connects low-level (color, edge and outline) and high-level (semantic) features of the image and color transform LUT features to generate appropriate adjustment weights. Meanwhile, we employ a LUT-Aware Module (LAM) to construct the channels and spatial attention for refining the LUT features. Since these modules have a more powerful representational capacity, they can better capture the intrinsic relationship between image semantics and LUT features. The extensive evaluations on standard benchmarks, including FiveK and HDR datasets, show that CANet achieves better performance compared to state-of-the-art methods.

## Framework
<img src="figures/framework.jpg" width="1024px"/>
![image](https://github.com/flyingbird93/CANet/assets/16755407/e9de400c-bf88-4f6d-b266-22bbcfd5ec56)


## Usage

### Requirements
Python3, requirements.txt

### Build
For pytorch 1.x:

    cd trilinear_cpp
    sh setup.sh

Please also replace the following lines:
```
# in image_adaptive_lut_train_paired.py, image_adaptive_lut_evaluation.py, demo_eval.py, and image_adaptive_lut_train_unpaired.py
from models import * --> from models_x import *
# in demo_eval.py
result = trilinear_(LUT, img) --> _, result = trilinear_(LUT, img)
# in image_adaptive_lut_train_paired.py and image_adaptive_lut_evaluation.py
combine_A = trilinear_(LUT,img) --> _, combine_A = trilinear_(LUT,img)
```

### Training
#### paired training
     python3 image_adaptive_lut_train_paired_with_cross_attention.py

### Evaluation
1. use python to generate and save the test images:

       python3 image_adaptive_lut_evaluation_with_cross_attention.py

speed can also be tested in above code.

2. use matlab to calculate the indexes used in our paper:

       average_psnr_ssim.m


### Tools
You can generate identity 3DLUT with arbitrary dimension by using `utils/generate_identity_3DLUT.py` as follows:

```
# you can replace 36 with any number you want
python3 utils/generate_identity_3DLUT.py -d 36
```


## Citation
```
@inproceedings{shirgb,
  title={RGB and LUT based Cross Attention Network for Image Enhancement},
  author={Shi, Tengfei and Chen, Chenglizhao and He, Yuanbo and Hao, Aimin},
  booktitle={34rd British Machine Vision Conference 2023, BMVC 2023, Aberdeen, UK, November 20-24},
  year={2023},
}

```
