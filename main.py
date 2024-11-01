# Importing Libraries
import torch
import torch.nn as nn

# Importing Classes
from feature_extractor import autoencoder_train
from helpful_functions import get_device
from data_processing import preprocessing_data, data_augmentation
from autoencoder import Autoencoder


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
    num_epochs = 120
    learning_rate = 0.00001

    autoencoder = Autoencoder(input_size, encoded_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_features, test_features = autoencoder_train(train_loader, test_loader, autoencoder, num_epochs, criterion, optimizer, input_size)


    # Classifier
    print("Step 3. Loading Classifier & Training & Validating")




if __name__ == '__main__':
    main()
