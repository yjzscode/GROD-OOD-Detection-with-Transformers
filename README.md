# **OOD-Detection-with-Transformers**
The implementation of our work *"How Out-of-Distribution Detection Learning Theory Enhances Transformer: Learnability and Reliability"*.

![framework](framework_v2.png)

## Introduction
Transformer networks excel in natural language processing and computer vision tasks. However, they still face challenges in generalizing to Out-of-Distribution (OOD) datasets, i.e. data whose distribution differs from that seen during training. The OOD detection aims to distinguish outliers while preserving in-distribution (ID) data performance. This paper introduces the OOD detection Probably Approximately Correct (PAC) Theory for transformers, which establishes the conditions for data distribution and model configurations for the learnability of transformers in terms of OOD detection. The theory demonstrates that outliers can be accurately represented and distinguished with sufficient data. The theoretical implications highlight the trade-off between theoretical principles and practical training paradigms. By examining this trade-off, we naturally derived the rationale for leveraging auxiliary outliers to enhance OOD detection. Our theory suggests that by penalizing the misclassification of outliers within the loss function and strategically generating soft synthetic outliers, one can robustly bolster the reliability of transformer networks. This approach yields a novel algorithm that ensures learnability and refines the decision boundaries between inliers and outliers. In practice, the algorithm consistently achieves state-of-the-art performance across various data formats.

:fire: For more information have a look at our paper (coming soon).

Authors: 

## News :new:

## Installation

The code has been tested with Python 3.8, CUDA 11.8, and pytorch 2.2.0+cu118. Any other version may require to update the code for compatibility.

## Data preparation
To download the data and display the data with the proper structure of folders, follow the instructions provided by [OpenOOD](https://github.com/Jingkang50/OpenOOD). Actually, this code of GROD follows the structure of OpenOOD and can be added to this elegant repository.

## Commands

### CV tasks
To run the fine-tuning:
```
cd ./OpenOOD_GROD/scripts/ood/grod
```
```
bash cifar100_train_grod.sh
```
To run the post-processing:
```
cd ./OpenOOD_GROD/scripts/ood/grod
```
```
bash cifar100_test_grod.sh
```

### NLP tasks
Run `OpenOOD_GROD/text_classification/easy_dev.ipynb` and `OpenOOD_GROD/text_classification/easy_dev_GPT.ipynb` for BERT and GPT-2 backbones respectively, and `OpenOOD_GROD/text_classification/Llama_Experiments` for Llama models.

### Learning how GROD narrows the gap between theory and reality
Codes of experiments for generated Gaussian mixture datasets are in the fold `Gaussian_distribution`.

## Citing our work
Please cite the following paper if you use our code:

## Acknowledgements


## TODOS
:soon: Add fine-tuned models

:white_check_mark: Update for the adaptability to large models and multi-class datasets

:soon: Report results for more benchmarks

