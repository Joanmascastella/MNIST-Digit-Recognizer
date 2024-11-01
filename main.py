# Importing Libraries
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing Classes
from feature_extractor import autoencoder_train
from helpful_functions import get_device
from data_processing import preprocessing_data, data_augmentation
from autoencoder import Autoencoder
from classifiers import *


def main():

    # Defining Data Paths
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    submission_file = "./data/sample_submission.csv"

    # Creating Instance of Device
    device = get_device()

    # Processing Data
    print("Step 1. Processing Data")
    train_processed, y_processed, test_processed = preprocessing_data(train_file, test_file)
    train_loader, test_loader = data_augmentation(train_processed, y_processed, test_processed)

    # Autoencoder
    print("Step 2. Loading And Running Autoencoder")
    input_size = 28 * 28
    encoded_size = 64
    num_epochs = 95
    learning_rate = 0.00001

    autoencoder = Autoencoder(input_size, encoded_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_features, test_features = autoencoder_train(train_loader, test_loader, autoencoder, num_epochs, criterion, optimizer, input_size)

    # Convert PyTorch tensors to numpy arrays for scikit-learn compatibility
    train_features_np = train_features.numpy()
    y_processed_np = y_processed.numpy()

    # Step 3: Split train_features into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_features_np, y_processed_np, test_size=0.2, random_state=42)

    # Step 4: Initialize and Train Classifiers
    classifiers = {
        "SVM": svm_classifier(),
        "Random Forest": rf_classifier(),
        "Gradient Boosting": gradient_boosting_classifier(),
        "k-NN": knn_classifier(),
        "MLP": mlp_classifier()
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"Training {name} classifier...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")

    # Step 5: Print and Compare Results
    print("\nClassifier Accuracy Comparison:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")


if __name__ == '__main__':
    main()
