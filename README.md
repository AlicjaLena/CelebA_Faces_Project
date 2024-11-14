# CelebA_Faces_Project

# Project Purpose:
The project focuses on loading and processing the CelebA dataset, which contains over 200,000 celebrity images, each annotated with 40 attributes. The task is to classify a specific attribute (e.g., "Smiling") using a custom neural network model. The processing involves resizing images and normalizing pixel values. Data augmentation enhances the diversity of training data, improving the model's generalization on new test data. A pre-trained model like MobileNetV2 is employed as a feature extractor, and custom output layers are added to classify the chosen attribute.

# Installation Instructions:

# This project addresses two main challenges:

## 1. Avoiding NonMatchingChecksumError During Dataset Download
When downloading large datasets like CelebA directly via tensorflow_datasets or similar tools, users often encounter the NonMatchingChecksumError due to mismatch issues during the download process. This project eliminates this problem by allowing the dataset to be downloaded locally and accessed directly from disk. To use the dataset, download it manually and place it in the following directory:
[Insert Dataset Download Link Here]

## 2. Predicting Multiple Facial Attributes from CelebA Dataset
The project uses a CSV file with image labels to classify images into two categories: Smiling and Not Smiling. However, the dataset includes a comprehensive list of attributes, making the code adaptable for predicting other features. The dataset can be accessed via the following link: https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ




