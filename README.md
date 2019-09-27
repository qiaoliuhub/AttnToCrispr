# Overview

This project is intended to use deep learning models for Crispr-Cas on-target efficiency prediction and off target specificity prediction.

Below is the layout of the whole model.

## AttnToMismatch_CNN

<p align="center">
  <img src="pictures/New_attnToMismatch_CNN.png" width="900"/>
</p>

This model includes four components: 
* Embedding layer
* Transformer layer
* Convolutional neural network
* Fully connected layer

## AttnToCrispr_CNN

<p align="center">
  <img src="pictures/AttnToCrispr_CNN.png" width="900"/>
</p>

This model includes four components: 
* Embedding layer
* Transformer layer
* Convolutional neural network
* Fully connected layer

# Requirement

* keras
* tensorflow
* pytorch
* sklearn
* pandas
* numpy

# Usage
## Specify which data or model to use, such as cpf1 and cpf1_OT.

```
python ./attn_to_crispr.py <cellline>
```