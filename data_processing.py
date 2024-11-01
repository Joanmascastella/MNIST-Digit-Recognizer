import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def preprocessing_data(train_file, test_file):
    # Load Data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Split features and labels
    train_X = train_df.drop(columns=['label']) / 255.0
    train_Y = train_df['label']
    test_X = test_df / 255.0

    # Convert to tensors
    X = torch.tensor(train_X.values, dtype=torch.float32).reshape(-1, 1, 28, 28)
    y = torch.tensor(train_Y.values, dtype=torch.long)
    test = torch.tensor(test_X.values, dtype=torch.float32).reshape(-1, 1, 28, 28)

    return X, y, test


class MNISTDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def data_augmentation(train_data, y_data, test_data):
    # Define augmentation and preprocessing pipelines
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    preprocess_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create train and test datasets
    train_dataset = MNISTDataset(data=train_data, labels=y_data, transform=augmentation_transform)
    test_dataset = MNISTDataset(data=test_data, transform=preprocess_transform)

    # Return DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader