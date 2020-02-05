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
* skorch
* visdom

# Usage
## Specify which data or model to use, such as cpf1 and cpf1_OT.

```
python ./attn_to_crispr.py <data/model>
```
<data/model> could be K562/A549/NB4/cpf1/cpf1_OT/deepCrispr_OT

## Training new model with customized dataset

1. Organize dataset format as the example dataset in dataset/customized_Cas9_OT
2. Save the new dataset as dataset/customized_Cas9_OT/customized_Cas9_OT_data.csv
3. 
```
python flexible_OT_crispr.py customized_Cas9_OT
```
4. Optional:
Specify training-testing split methods:
change split_method in "models/customized_Cas9_OT/config.py":
* "regular" for n-fold split
* "stratified" for leave sgRNAs out split
