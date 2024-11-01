# Importing Libraries
import torch

# Importing Classes
from helpful_functions import get_device


def main():

    # Defining Data Paths
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    submission_file = "./data/sample_submission.csv"

    # Creating Instance of Device
    device = get_device()

    # Processing Data
    print("Step 1. Processing Data")



    # Autoencoder
    print("Step 2. Loading And Running Autoencoder")


    # Classifier
    print("Step 3. Loading Classifier & Training & Validating")




if __name__ == '__main__':
    main()
