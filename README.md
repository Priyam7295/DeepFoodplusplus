# Food101 Classification - Fine-Tuning VGG for Food Recognition

This repository contains the implementation of a food image classification model using the Food101 dataset. By extracting relevant features and fine-tuning a VGG-based architecture, this model achieves superior performance, surpassing the DeepFood model with a **Top-5 accuracy** of **96%** and **Top-1 accuracy** of **91%**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)


## Overview

This project focuses on image classification for food-related categories using the Food101 dataset. The approach used in this repository involves feature extraction followed by fine-tuning on a VGG-based architecture. The fine-tuned model was trained to classify food images into 101 categories and tested against the Food101 dataset.

### Key Highlights:
- **Model Architecture:** VGG (Fine-tuning)
- **Dataset:** Food101
- **Performance:**
  - **Top-1 Accuracy:** 81%
  - **Top-5 Accuracy:** 96%
  
## Dataset

The **Food101 dataset** consists of 101 food categories with 101,000 images, each representing a different food item. It is used as the benchmark dataset for training and evaluating the model.

- **Dataset Download Link:** [Food101 Dataset](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)

## Model Architecture

The model utilizes the VGG architecture as a base for fine-tuning. The steps include:

1. **Feature Extraction:** We first extract features using the pre-trained VGG model.
2. **Fine-Tuning:** The top layers are fine-tuned using the extracted features, allowing the model to adapt to the Food101 dataset for improved classification accuracy.

### VGG Architecture
- Base model: VGG16
- Fine-tuned for 101 food categories

### Results
1. Top-1 Accuracy: 81% (approx.)
2. Top-5 Accuracy: 96% (approx.)


