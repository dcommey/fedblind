# data/data_loader.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import dirichlet
from utils.config import MNIST_DIR, CIFAR10_DIR, CIFAR100_DIR, SVHN_DIR, HAR_DIR
import os
import logging
from urllib.request import urlretrieve
import zipfile
import shutil
import random
import time

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        
        # Ensure the image is a float tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # If it's already a tensor, make sure it's float
        img = img.float()
        
        # Normalize if not done by transform
        if img.max() > 1:
            img = img / 255.0
        
        return img

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=MNIST_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=MNIST_DIR, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=CIFAR10_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=CIFAR10_DIR, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = datasets.CIFAR100(root=CIFAR100_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=CIFAR100_DIR, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_svhn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.SVHN(root=SVHN_DIR, split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root=SVHN_DIR, split='test', download=True, transform=transform)
    return train_dataset, test_dataset

def load_har():
    har_train_file = os.path.join(HAR_DIR, "har_train.csv")
    har_test_file = os.path.join(HAR_DIR, "har_test.csv")

    try:
        if not (os.path.exists(har_train_file) and os.path.exists(har_test_file)):
            download_and_prepare_har()

        # Create unique column names
        train_columns = [f"{x}_{i}" for i, x in enumerate(pd.read_csv(har_train_file, nrows=0).columns)]
        test_columns = [f"{x}_{i}" for i, x in enumerate(pd.read_csv(har_test_file, nrows=0).columns)]

        # Load the data with unique column names
        train_data = pd.read_csv(har_train_file, header=0, names=train_columns)
        test_data = pd.read_csv(har_test_file, header=0, names=test_columns)
        
        X_train = train_data.drop('Activity_561', axis=1).values
        y_train = train_data['Activity_561'].values
        X_test = test_data.drop('Activity_561', axis=1).values
        y_test = test_data['Activity_561'].values
        
        # Normalize the data
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_normalized), torch.LongTensor(y_train))
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test_normalized), torch.LongTensor(y_test))
        
        return train_dataset, test_dataset
    except Exception as e:
        logging.error(f"Error loading HAR dataset: {str(e)}")
        raise

def download_and_prepare_har():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = os.path.join(HAR_DIR, "har_dataset.zip")
    extract_path = os.path.join(HAR_DIR, "UCI HAR Dataset")

    try:
        # Create HAR_DIR if it doesn't exist
        os.makedirs(HAR_DIR, exist_ok=True)

        # Download the dataset
        logging.info("Downloading HAR dataset...")
        urlretrieve(url, zip_path)

        # Extract the dataset
        logging.info("Extracting HAR dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(HAR_DIR)

        # Prepare the dataset
        logging.info("Preparing HAR dataset...")
        features = pd.read_csv(os.path.join(extract_path, 'features.txt'), sep=r'\s+', header=None, names=['index', 'feature'])
        feature_names = [f"feature_{i}" for i in range(len(features))]  # Create unique feature names

        def load_dataset(subset):
            X = pd.read_csv(os.path.join(extract_path, subset, f'X_{subset}.txt'), sep=r'\s+', header=None, names=feature_names)
            y = pd.read_csv(os.path.join(extract_path, subset, f'y_{subset}.txt'), sep=r'\s+', header=None, names=['Activity'])
            return pd.concat([X, y], axis=1)

        train_data = load_dataset('train')
        test_data = load_dataset('test')

        # Save prepared datasets
        train_data.to_csv(os.path.join(HAR_DIR, "har_train.csv"), index=False)
        test_data.to_csv(os.path.join(HAR_DIR, "har_test.csv"), index=False)

        # Clean up
        logging.info("Cleaning up temporary files...")
        os.remove(zip_path)
        
        # Attempt to remove the extracted directory
        try:
            shutil.rmtree(extract_path)
        except PermissionError:
            logging.warning(f"Unable to remove directory: {extract_path}. It may be in use.")
            logging.warning("Please manually delete this directory if it's no longer needed.")

        logging.info("HAR dataset prepared successfully.")
    except Exception as e:
        logging.error(f"Error preparing HAR dataset: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            logging.warning(f"Extracted data remains in: {extract_path}")
        raise

def create_unlabeled_subset(dataset, unlabeled_ratio=0.3):
    num_samples = len(dataset)
    num_unlabeled = int(num_samples * unlabeled_ratio)
    unlabeled_indices = np.random.choice(num_samples, num_unlabeled, replace=False)
    labeled_indices = np.setdiff1d(np.arange(num_samples), unlabeled_indices)
    
    labeled_data = Subset(dataset, labeled_indices)
    
    # Check if the dataset has a 'data' attribute (like MNIST, CIFAR)
    if hasattr(dataset, 'data'):
        if isinstance(dataset.data, np.ndarray):
            unlabeled_data = dataset.data[unlabeled_indices]
        elif isinstance(dataset.data, torch.Tensor):
            unlabeled_data = dataset.data[unlabeled_indices].numpy()
        else:
            # For datasets with PIL Images
            unlabeled_data = [dataset.data[i] for i in unlabeled_indices]
    else:
        # For datasets without 'data' attribute (like HAR), create a new dataset
        unlabeled_data = [dataset[i][0] for i in unlabeled_indices]
    
    # Create transform that includes ToTensor and Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    unlabeled_dataset = UnlabeledDataset(unlabeled_data, transform=transform)
    
    return labeled_data, unlabeled_dataset

def dirichlet_split_noniid(labels, alpha, n_clients):
    """
    Splits data among clients using Dirichlet distribution.
    Args:
        labels: tensor of labels
        alpha: parameter for Dirichlet distribution
        n_clients: number of clients to split data between
    """
    n_clients = int(n_clients)
    if n_clients <= 0:
        raise ValueError(f"Number of clients must be positive, got {n_clients}")
        
    n_classes = len(torch.unique(labels))
    if n_classes == 0:
        raise ValueError("No classes found in labels")
        
    # Initialize empty lists for each client
    client_idcs = [[] for _ in range(n_clients)]
    
    # For each class
    for k in range(n_classes):
        # Get indices of samples from this class
        idx_k = torch.where(labels == k)[0]
        
        if len(idx_k) == 0:  # Skip if no samples for this class
            continue
            
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        
        # Convert to number of samples
        proportions = (proportions * len(idx_k)).astype(int)
        
        # Fix rounding errors
        proportions[-1] = len(idx_k) - proportions[:-1].sum()
        
        # Split indices according to proportions
        start_idx = 0
        for client_id, prop in enumerate(proportions):
            if prop > 0:  # Only add if there are samples to add
                client_idcs[client_id].extend(idx_k[start_idx:start_idx + prop].tolist())
                start_idx += prop
    
    return client_idcs

def get_dataloader(dataset_name, num_clients, unlabeled_ratio=0.3, alpha=0.5, batch_size=32, labeled_subset_size=1000):
    """
    Get dataloaders for federated learning.
    Returns:
        tuple: (client_data, client_labels, test_loader, unlabeled_loader, labeled_subset_loader)
    """
    try:
        if dataset_name == 'mnist':
            train_dataset, test_dataset = load_mnist()
        elif dataset_name == 'cifar10':
            train_dataset, test_dataset = load_cifar10()
        elif dataset_name == 'cifar100':
            train_dataset, test_dataset = load_cifar100()
        elif dataset_name == 'svhn':
            train_dataset, test_dataset = load_svhn()
        elif dataset_name == 'har':
            train_dataset, test_dataset = load_har()
        else:
            raise ValueError("Invalid dataset name")
    except Exception as e:
        logging.error(f"Error loading {dataset_name} dataset: {str(e)}")
        return None, None, None

    labeled_data, unlabeled_data = create_unlabeled_subset(train_dataset, unlabeled_ratio)
    
    # Create labeled subset
    num_classes = len(set(y for _, y in labeled_data))
    labeled_subset = get_subset_of_labeled_data(labeled_data, labeled_subset_size, num_classes)
    
    client_idcs = dirichlet_split_noniid(torch.tensor([y for _, y in labeled_data]), alpha, num_clients)
    client_dataloaders = []
    for idcs in client_idcs:
        if len(idcs) == 0:
            # Create an empty dataset to avoid errors
            empty_dataset = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
            loader = DataLoader(empty_dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = DataLoader(Subset(labeled_data, idcs), batch_size=batch_size, shuffle=True)
        client_dataloaders.append(loader)
    
    # Similarly, for labeled_subset_loader, if its indices are empty, return an empty DataLoader
    if labeled_subset_size == 0:
        empty_dataset = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
        labeled_subset_loader = DataLoader(empty_dataset, batch_size=batch_size, shuffle=True)
    else:
        labeled_subset_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True)
    
    return client_dataloaders, test_loader, unlabeled_loader, labeled_subset_loader

def get_subset_of_labeled_data(dataset, num_samples, num_classes):
    """
    Get a balanced subset of labeled data from the given dataset.
    
    Args:
    dataset (torch.utils.data.Dataset): The full dataset to sample from.
    num_samples (int): Total number of samples to retrieve.
    num_classes (int): Number of classes in the dataset.
    
    Returns:
    torch.utils.data.Subset: A subset of the original dataset with balanced classes.
    """
    samples_per_class = num_samples // num_classes
    class_counts = [0] * num_classes
    subset_indices = []

    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    for idx in all_indices:
        _, label = dataset[idx]
        if class_counts[label] < samples_per_class:
            subset_indices.append(idx)
            class_counts[label] += 1
        
        if sum(class_counts) == num_samples:
            break

    return Subset(dataset, subset_indices)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client_loaders, test_loader, unlabeled_loader, labeled_subset_loader = get_dataloader('mnist', batch_size=64, num_clients=5, alpha=0.5, unlabeled_ratio=0.3)
    
    if client_loaders is not None:
        print(f"Number of client dataloaders: {len(client_loaders)}")
        print(f"Samples in first client: {len(client_loaders[0].dataset)}")
        print(f"Samples in test set: {len(test_loader.dataset)}")
        print(f"Samples in unlabeled set: {len(unlabeled_loader.dataset)}")
        
        # Visualize a batch from unlabeled data
        import matplotlib.pyplot as plt
        
        unlabeled_iter = iter(unlabeled_loader)
        images = next(unlabeled_iter)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("Failed to load dataset.")