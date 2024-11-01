import torch
from helpful_functions import get_device

device = get_device()

def autoencoder_train(train_loader, test_loader, autoencoder, num_epochs, criterion, optimizer, input_size):
    # Train the autoencoder
    for epoch in range(num_epochs):
        for images, _ in train_loader:
            images = images.view(-1, input_size).to(device)
            _, decoded = autoencoder(images)
            loss = criterion(decoded, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 epochs
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # Extract features for train data after training completes
    train_features = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.view(-1, input_size).to(device)
            encoded, _ = autoencoder(images)
            train_features.append(encoded.cpu())
    train_features = torch.cat(train_features)

    # Extract features for test data after training completes
    test_features = []
    with torch.no_grad():
        for images in test_loader:
            images = images.view(-1, input_size).to(device)
            encoded, _ = autoencoder(images)
            test_features.append(encoded.cpu())
    test_features = torch.cat(test_features)

    return train_features, test_features