# # CelebA Dataset Analysis with MobileNetV2

### About the Project
This project is my **first portfolio project** as an aspiring developer, where I aimed to showcase my ability to solve practical challenges and implement machine learning solutions. Specifically, I focused on:

1. **Solving the problem of downloading the CelebA dataset**: By providing a local setup, I addressed issues like `NonMatchingChecksumError` that often occur during automatic dataset downloads.

2. **Using a pre-trained MobileNetV2 model**: The project demonstrates how to leverage a pre-trained model for image analysis, customizing it to classify facial attributes such as "Smiling" or "Not Smiling" from the CelebA dataset.

I hope this project proves useful for others exploring machine learning and computer vision, and serves as a helpful example of solving common data challenges while building adaptable solutions. Thank you for your time and interest! 😊

# Project Purpose:
The project focuses on loading and processing the CelebA dataset, which contains over 200,000 celebrity images, each annotated with 40 attributes. The task is to classify a specific attribute (e.g., "Smiling") using a custom neural network model. The processing involves resizing images and normalizing pixel values. Data augmentation enhances the diversity of training data, improving the model's generalization on new test data. A pre-trained model like MobileNetV2 is employed as a feature extractor, and custom output layers are added to classify the chosen attribute.

# This project addresses two main challenges:

## 1. Avoiding NonMatchingChecksumError During Dataset Download and failed to download CelebA dataset using download=True 
When downloading large datasets like CelebA directly via tensorflow_datasets or similar tools, users often encounter the NonMatchingChecksumError due to mismatch issues during the download process. This project eliminates this problem by allowing the dataset to be downloaded locally and accessed directly from disk. To use the dataset, dowload it directly from the authors' google drive: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

## 2. Predicting Multiple Facial Attributes from CelebA Dataset
The project uses a CSV file with image labels to classify images into two categories: Smiling and Not Smiling. However, the dataset includes a comprehensive list of attributes, making the code adaptable for predicting other features. The dataset can be accessed via the following link: https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ


# Installation Instructions:
1. 



