# ViFR


This repository contains the training codes for the paper  "***ViFR: Visual Feature Reﬁnement for Zero-Shot Learning***". A preliminary conference version of this work appears in the Proceedings of the ICCV'21, titled ["FREE: Feature Refinement for Generalized Zero-Shot Learning"](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_FREE_Feature_Refinement_for_Generalized_Zero-Shot_Learning_ICCV_2021_paper.pdf).

<!-- ![](figs/ViFR.png) -->


## Running Environment
The implementation of **ViFR** is mainly based on Python 3.8.8 and [PyTorch](https://pytorch.org/) 1.8.0. To install all required dependencies:
```
$ pip install -r requirements.txt
```
Additionally, we use [Weights & Biases](https://wandb.ai/site) (W&B) to keep track and organize the results of experiments. You may need to follow the [online documentation](https://docs.wandb.ai/quickstart) of W&B to quickstart. To run these codes, [sign up](https://app.wandb.ai/login?signup=true) an online account to track experiments or create a [local wandb server](https://hub.docker.com/r/wandb/local) using docker (recommended).

## Download Dataset 

We trained the model on three popular ZSL benchmarks: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html) and [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip). In order to train the **ViFR**, you should firstly download these datasets as well as the xlsa17. Then decompress and organize them as follows: 
```
.
├── data
│   ├── CUB/CUB_200_2011/...
│   ├── SUN/images/...
│   ├── AWA2/Animals_with_Attributes2/...
│   └── xlsa17/data/...
└── ···
```


## Visual Features Preprocessing

In this step, you should run the following commands to extract the visual features of three datasets:

```
$ python preprocessing.py --dataset CUB --compression
$ python preprocessing.py --dataset SUN --compression
$ python preprocessing.py --dataset AWA2 --compression
```

## Training ViFR from Scratch
In `./wandb_config`, we provide our parameters setting of conventional ZSL (CZSL) and generalized ZSL (GZSL) tasks for CUB, SUN, and AWA2. You can run the following commands to train the **ViFR** from scratch:

```
$ python train_ViFR_CUB.py
$ python train_ViFR_SUN.py 
$ python train_ViFR_AWA2.py 
```
**Note**: Please load the corresponding setting when aiming at the CZSL task.

## Results
We also provide trained models ([~~Google Drive~~ the download link will be released as soon as possible]()) on three datasets. You can download these `.pth` files and validate the results in our paper. Please refer to the [test branch]() for testing codes and usage.
Following table shows the results of our released models using various evaluation protocols on three datasets, both in the CZSL and GZSL settings:

**The input size of ResNet-101 is 224x224:**
| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 69.1 | 57.8 | 62.7 | 60.1 |
| SUN | 65.6 | 48.8 | 35.2 | 40.9 |
| AWA2 | 73.7 | 58.4 | 81.4 | 68.0 |

**The input size of ResNet-101 is 448x448:**
| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 74.5 | 63.9 | 72.0 | 67.7 |
| SUN | 69.2 | 51.3 | 40.0 | 44.7 |
| AWA2 | 77.8 | 68.2 | 78.9 | 73.2 |

**Note**:  The training of our models and all of above results are run on a server with an AMD Ryzen 7 5800X CPU, 128GB memory and a NVIDIA RTX A6000 GPU (48GB).


## References
Parts of our codes based on:
* [hbdat/cvpr20_DAZLE](https://github.com/hbdat/cvpr20_DAZLE)
* [shiming-chen/FREE](https://github.com/shiming-chen/FREE)
* [akshitac8/tfvaegan](https://github.com/akshitac8/tfvaegan)

## Contact
If you have any questions about codes, please don't hesitate to contact us by gchenshiming@gmail.com or hoongzm@gmail.com.
