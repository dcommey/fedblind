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
        
        # Convert to PyTorch tensor directly if it's tabular data (like HAR)
        if len(img.shape) == 1:
            # Use clone().detach() instead of torch.tensor() to avoid warning
            if isinstance(img, torch.Tensor):
                return img.clone().detach().float()
            else:
                return torch.tensor(img, dtype=torch.float32)
        
        # Handle SVHN and other image datasets differently
        if isinstance(img, np.ndarray):
            # For SVHN which has weird shapes
            if img.shape == (1, 1, 32) or len(img.shape) > 2:
                # Reshape to a format PIL can handle
                if img.shape[0] == 1:  # Handle single-channel case
                    img = img.reshape(img.shape[1:])
                
                # If it's RGB but in a weird order, transpose it
                if len(img.shape) == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                
                # Ensure uint8 format for PIL
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
            
            # Convert to PIL Image if it's a 2D or 3D array (image)
            if len(img.shape) >= 2:
                try:
                    img = Image.fromarray(img)
                except TypeError:
                    # If conversion fails, use tensor directly
                    img = torch.tensor(img, dtype=torch.float32)
                    if self.transform:
                        img = self.transform(img)
                    return img

        # Apply transforms if any
        if self.transform and isinstance(img, Image.Image):
            img = self.transform(img)
        
        # Ensure the image is a float tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Make sure tensor is float
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
        
        # Convert labels to be zero-indexed (HAR labels are 1-6, we need 0-5)
        y_train = y_train - 1
        y_test = y_test - 1
        
        # Ensure there are no out-of-bounds values
        assert np.all(y_train >= 0) and np.all(y_train < 6), "Train labels should be 0-5"
        assert np.all(y_test >= 0) and np.all(y_test < 6), "Test labels should be 0-5"
        
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
        features = pd.read_csv(os.path.join(extract_path, 'features.txt'), sep='\s+', header=None, names=['index', 'feature'])
        feature_names = [f"feature_{i}" for i in range(len(features))]  # Create unique feature names

        def load_dataset(subset):
            X = pd.read_csv(os.path.join(extract_path, subset, f'X_{subset}.txt'), sep='\s+', header=None, names=feature_names)
            y = pd.read_csv(os.path.join(extract_path, subset, f'y_{subset}.txt'), sep='\s+', header=None, names=['Activity'])
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
    """Create labeled and unlabeled subsets from a dataset."""
    num_samples = len(dataset)
    num_unlabeled = int(num_samples * unlabeled_ratio)
    unlabeled_indices = np.random.choice(num_samples, num_unlabeled, replace=False)
    labeled_indices = np.setdiff1d(np.arange(num_samples), unlabeled_indices)
    
    labeled_data = Subset(dataset, labeled_indices)
    
    # Handle HAR and other non-image datasets
    if hasattr(dataset, 'data') and isinstance(dataset.data, np.ndarray):
        # For HAR or other tabular datasets with a data attribute
        unlabeled_data = dataset.data[unlabeled_indices]
    elif hasattr(dataset, 'data') and isinstance(dataset.data, torch.Tensor):
        # For datasets with tensor data
        unlabeled_data = dataset.data[unlabeled_indices].numpy()
    elif hasattr(dataset, 'data'):
        # For datasets with other data formats
        try:
            unlabeled_data = np.array([dataset.data[i] for i in unlabeled_indices])
        except:
            # For datasets where direct indexing doesn't work
            unlabeled_data = [dataset[i][0] for i in unlabeled_indices]
    else:
        # For datasets without a 'data' attribute, use dataset[i][0]
        unlabeled_data = [dataset[i][0] for i in unlabeled_indices]
    
    # Use dataset-specific transforms
    transform = None
    if hasattr(dataset, 'transform') and dataset.transform is not None:
        transform = dataset.transform
    
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
            # HAR needs a smaller subset size due to fewer samples
            labeled_subset_size = min(labeled_subset_size, len(train_dataset) // 2)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
    except Exception as e:
        logging.error(f"Error loading {dataset_name} dataset: {str(e)}")
        raise

    labeled_data, unlabeled_data = create_unlabeled_subset(train_dataset, unlabeled_ratio)
    
    # Determine number of classes for the dataset
    if dataset_name == 'mnist' or dataset_name == 'cifar10' or dataset_name == 'svhn':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'har':
        num_classes = 6
    else:
        num_classes = 10  # Default
    
    # Create labeled subset with error handling
    try:
        labeled_subset = get_subset_of_labeled_data(labeled_data, labeled_subset_size, num_classes)
        if len(labeled_subset) == 0:
            logging.warning(f"Empty labeled subset for {dataset_name}, using first sample")
            labeled_subset = Subset(labeled_data, [0])  # Include at least one sample
    except Exception as e:
        logging.error(f"Error creating labeled subset: {str(e)}")
        labeled_subset = Subset(labeled_data, [0])  # Fallback to single sample
    
    # Create client datasets with Dirichlet distribution
    try:
        client_idcs = dirichlet_split_noniid(
            torch.tensor([labeled_data[i][1] for i in range(len(labeled_data))]), 
            alpha, 
            num_clients
        )
        client_dataloaders = [
            DataLoader(Subset(labeled_data, idcs), batch_size=batch_size, shuffle=True) 
            for idcs in client_idcs
        ]
    except Exception as e:
        logging.error(f"Error splitting data among clients: {str(e)}")
        # Fallback to random splitting
        indices = list(range(len(labeled_data)))
        random.shuffle(indices)
        client_size = len(indices) // num_clients
        client_dataloaders = [
            DataLoader(
                Subset(labeled_data, indices[i*client_size:(i+1)*client_size]),
                batch_size=batch_size, 
                shuffle=True
            ) 
            for i in range(num_clients)
        ]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True)
    labeled_subset_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
    
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
    # Special handling for HAR dataset which may have fewer samples
    if len(dataset) < num_samples:
        return dataset  # Return the entire dataset if smaller than requested
        
    # Ensure num_samples is at least equal to the number of classes
    samples_per_class = max(1, num_samples // num_classes)
    
    # Get all class labels from the dataset
    all_labels = []
    for i in range(min(1000, len(dataset))):  # Sample up to 1000 items to determine classes
        _, label = dataset[i]
        all_labels.append(label)
    
    # Find actual number of unique classes
    unique_labels = set(all_labels)
    actual_num_classes = len(unique_labels)
    
    if actual_num_classes == 0:
        raise ValueError("No classes found in dataset")
    
    # Adjust samples per class based on actual classes found
    samples_per_class = max(1, num_samples // actual_num_classes)
    
    # Initialize counters and result indices
    class_counts = {label: 0 for label in unique_labels}
    subset_indices = []

    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    for idx in all_indices:
        _, label = dataset[idx]
        # Convert tensor to scalar if needed
        if isinstance(label, torch.Tensor):
            label = label.item() if label.numel() == 1 else label[0].item()
            
        if label in class_counts and class_counts[label] < samples_per_class:
            subset_indices.append(idx)
            class_counts[label] += 1
        
        # Stop when we have enough samples or all classes are satisfied
        if sum(class_counts.values()) >= num_samples or all(count >= samples_per_class for count in class_counts.values()):
            break

    # Ensure we have at least one sample
    if not subset_indices:
        subset_indices = [0]  # Include at least one sample to avoid empty dataset
        
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