# MNIST Digit Recognizer with Hyperparameter Optimization

This project involves building, training, and optimizing multiple machine learning classifiers to recognize handwritten digits in the MNIST dataset. Through a detailed hyperparameter tuning process and implementation improvements, I advanced my standing on Kaggle, moving from position 1037 to 837, putting me in the top 57% with an accuracy of 97.799%.

## Project Overview

This project consists of several stages, from data preprocessing and augmentation to model training and evaluation. The main steps include:
1. **Data Processing**: The MNIST dataset is loaded and normalized.
2. **Data Augmentation**: Training images undergo random rotations and affine transformations.
3. **Classifier Training**: Multiple classifiers are trained, with hyperparameter tuning using `RandomizedSearchCV`.
4. **Evaluation and Submission**: The best-performing model predictions are formatted and saved for Kaggle submission.

## Classifiers Used

1. **Support Vector Machine (SVM)**: Using optimized `SVC` with hyperparameter tuning.
2. **Random Forest Classifier**: Optimized with parameters such as `n_estimators`, `max_depth`, and `min_samples_split`.
3. **Gradient Boosting (XGBoost)**: Tuned with various parameters, including `learning_rate` and `max_depth`.
4. **k-Nearest Neighbors (k-NN)**: Using a variety of `n_neighbors` values and distance metrics.
5. **Multi-Layer Perceptron (MLP)**: Tuned with different layer sizes, `learning_rate_init`, and regularization.

Each classifier’s performance is evaluated on the validation dataset, and the test predictions are saved in a CSV format for each model.

## Accuracy and Ranking Improvement

After implementing various optimizations, I achieved an accuracy of **97.799%**, moving up from rank **1037** to **837** on the Kaggle leaderboard, placing me in the top **57%**.

## Structure

•	data/: Contains the dataset files and submission CSV files.

•	main.py: Main script that orchestrates data loading, model training, and submission generation.

•	classifiers.py: Contains functions to initialize classifiers with hyperparameter tuning.

## Requirements

To run this project, you need the following Python packages:

```plaintext
pandas
torch
torchvision
scikit-learn
xgboost
tqdm
