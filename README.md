# Automated-Skin-Cancer-Detection-Using-EfficientNetB3


## Project Overview

This project focuses on the automated detection of skin cancer using Deep Learning and Transfer Learning techniques. The system utilizes the EfficientNet-B3 architecture to classify dermoscopic skin lesion images as either benign or malignant, assisting in the early diagnosis of skin cancer.

## Objectives

* Develop an automated skin cancer detection system.
* Improve diagnostic accuracy using Deep Learning.
* Assist healthcare professionals in early disease detection.
* Reduce manual effort and support clinical decision-making.

## Dataset

The project uses a dermoscopic image dataset containing skin lesion images categorized into:

* Benign Skin Lesions
* Malignant Skin Lesions

The dataset is preprocessed and augmented to improve model performance and generalization.

## Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNet-B3
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Scikit-learn
* Jupyter Notebook

## Project Workflow

1. Data Collection
2. Image Preprocessing
3. Data Augmentation
4. Transfer Learning using EfficientNet-B3
5. Model Training and Fine-Tuning
6. Model Evaluation
7. Skin Cancer Prediction

## Model Architecture

EfficientNet-B3 was used as the base model due to its high accuracy and computational efficiency. Transfer learning was applied by utilizing pre-trained ImageNet weights and fine-tuning the network on the skin lesion dataset.

## Performance Metrics

* Accuracy: 94%
* AUC Score: 0.99
* Precision
* Recall
* F1-Score

## Results

The proposed model achieved 94% classification accuracy and an AUC score of 0.99, demonstrating excellent performance in distinguishing between benign and malignant skin lesions. The system can support early diagnosis and improve healthcare outcomes.

## Project Structure

```text
Automated-Skin-Cancer-Detection-Using-EfficientNetB3/
│
├── dataset/
├── models/
├── notebooks/
├── images/
├── requirements.txt
├── README.md
└── Skin_Cancer_Detection.ipynb
```

## Future Enhancements

* Multi-class skin disease classification.
* Real-time web application deployment.
* Integration with mobile healthcare applications.
* Explainable AI techniques for better clinical interpretation.

