# Dermatologist AI: Skin Cancer Detection with Convolutional Neural Networks (CNNs)

This repository contains the definition and evaluation of a Convolutional Neural Network (CNN) which aims to classify skin lesion images into three categories:

- [Melanoma](https://es.wikipedia.org/wiki/Melanoma): malign cancer, one of the deadliest.
- [Nevus](https://en.wikipedia.org/wiki/Nevus): benign skin lesion (mole or birthmark).
- [Seborrheic keratosis](https://en.wikipedia.org/wiki/Seborrheic_keratosis): benign skin tumor.

![Skin Disease Classes](./images/skin_disease_classes.png)

The motivation comes from the Nature paper by [Esteva et al.](https://www.nature.com/articles/nature21056.epdf?author_access_token=8oxIcYWf5UNrNpHsUHd2StRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuPakXos4UhQAFZ750CsBNMMsISFHIKinKDMKjShCpHIlYPYUHhNzkn6pSnOCt0Ftf6), in which the authors show how a CNN architecture based on the [Inception-V3](https://en.wikipedia.org/wiki/Inceptionv3) network achieves a **dermatologist-level classification of skin cancer**.

I forked the starter code from the Github repository [udacity/dermatologist-ai](https://github.com/udacity/dermatologist-ai). The dataset is from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://challenge.isic-archive.com/landing/2017/).

## ISIC Challenge 2017

Although the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://challenge.isic-archive.com/landing/2017/) is already closed, information on the challenge can be obtained from the official [website](https://challenge.isic-archive.com/landing/2017/).

The challenge had three parts or tasks:

- Part 1: Lesion Segmentation Task
- Part 2: Dermoscopic Feature Classification Task
- Part 3: Disease Classification Task

The present project deals only with the third part or task.

### Challenge Results

The challenge organizers published an interesting summary of the results and insights from the best contributions in [this article](https://arxiv.org/pdf/1710.05006.pdf); some interesting points associated with the third part/task are:

- A
- B
- C

Some of the works that obtained the best results are: 

- Matsunaga K, Hamada A, Minagawa A, Koga H. [“Image Classification of Melanoma, Nevus and Seborrheic Keratosis by Deep Neural Network Ensemble”](https://arxiv.org/ftp/arxiv/papers/1703/1703.03108.pdf). 
- Díaz IG. [“Incorporating the Knowledge of Dermatologists to Convolutional Neural Networks for the Diagnosis of Skin Lesions”](https://arxiv.org/pdf/1703.01976.pdf). [**Code**](https://github.com/igondia/matconvnet-dermoscopy).
- Menegola A, Tavares J, Fornaciali M, Li LT, Avila S, Valle E. [“RECOD Titans at ISIC Challenge 2017”](https://arxiv.org/abs/1703.04819). [**Code**](https://github.com/learningtitans/isbi2017-part3).

### Dataset

I downloaded the dataset from the links provided by Udacity to the non-committed folder `data/`, which is subdivided in the train, validation and test subfolders as well as class-name subfolders:

- [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip) (5.3 GB)
- [validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip) (824.5 MB)
- [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) (5.1 GB)

## Overview and File Structure

TBD.

```
Instructions.md             # Original project instructions from Udacity
LICENSE.txt                 # License
README.md                   # Current file
get_results.py              #
ground_truth.csv            #
images/                     #
requirements.txt            # Dependencies
sample_predictions.csv      # 
```
### How to Use This

TBD.

### Dependencies

TBD.

## Skin Cancer Detection Model

TBD.

## Preliminary Results

TBD.

## Possible Improvements

TBD.

## Authorship and License

This repository was forked from [udacity/dermatologist-ai](https://github.com/udacity/dermatologist-ai) and modified to the present status following the original [license](LICENSE.txt) from Udacity.

Mikel Sagardia, 2022.  
No guarantees.

