# Importing Libraries
import torch
import pandas as pd
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Importing Classes
from helpful_functions import get_device
from data_processing import preprocessing_data, data_augmentation
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

    # Reshape data for compatibility with scikit-learn classifiers
    X_data = train_processed.view(train_processed.size(0), -1).numpy()  # Flatten images to 1D arrays
    y_data = y_processed.numpy()
    X_test = test_processed.view(test_processed.size(0), -1).numpy()  # Flatten test data as well

    # Step 3: Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

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

        start_time = time.time()

        with tqdm(total=1, desc=f"Training {name}", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            clf.fit(X_train, y_train)
            pbar.update(1)

        end_time = time.time()
        duration = end_time - start_time
        print(f"{name} training completed in {duration:.2f} seconds.")

        # Validation
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")

        # Step 6: Generate predictions on test data
        test_predictions = clf.predict(X_test)

        # Load sample submission file and add predictions
        submission = pd.read_csv(submission_file)
        submission['Label'] = test_predictions

        # Save predictions to a new CSV file for each classifier
        submission_output_file = f"./data/{name}_submission.csv"
        submission.to_csv(submission_output_file, index=False)
        print(f"Submission file saved for {name} classifier as {submission_output_file}")

    # Step 7: Print and Compare Results
    print("\nClassifier Accuracy Comparison:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

if __name__ == '__main__':
    main()