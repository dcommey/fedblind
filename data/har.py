import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from .data_loader import create_dirichlet_distributed_data
from utils.config import HAR_DIR

HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
HAR_FOLDER = os.path.join(HAR_DIR, "UCI HAR Dataset")

def download_and_extract_har():
    if not os.path.exists(HAR_FOLDER):
        print("Downloading HAR dataset...")
        os.makedirs(HAR_DIR, exist_ok=True)
        zip_path = os.path.join(HAR_DIR, "har_dataset.zip")
        urllib.request.urlretrieve(HAR_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(HAR_DIR)
        os.remove(zip_path)
        print("HAR dataset downloaded and extracted.")

def load_file(filepath):
    return pd.read_csv(filepath, header=None, delim_whitespace=True).values

def load_group(filenames, prefix=''):
    loaded = [load_file(os.path.join(prefix, name)) for name in filenames]
    return np.dstack(loaded)

def load_dataset_group(group, prefix=''):
    filepath = os.path.join(prefix, group, 'Inertial Signals')
    filenames = [
        f'total_acc_x_{group}.txt', f'total_acc_y_{group}.txt', f'total_acc_z_{group}.txt',
        f'body_acc_x_{group}.txt', f'body_acc_y_{group}.txt', f'body_acc_z_{group}.txt',
        f'body_gyro_x_{group}.txt', f'body_gyro_y_{group}.txt', f'body_gyro_z_{group}.txt'
    ]
    X = load_group(filenames, filepath)
    y = load_file(os.path.join(prefix, group, f'y_{group}.txt'))
    return X, y

def load_har(num_clients, alpha):
    download_and_extract_har()
    
    x_train, y_train = load_dataset_group('train', HAR_FOLDER)
    x_test, y_test = load_dataset_group('test', HAR_FOLDER)
    
    y_train = y_train - 1
    y_test = y_test - 1
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    clients_data = create_dirichlet_distributed_data(x_train, y_train, num_clients, alpha)
    
    return clients_data, x_test, y_test