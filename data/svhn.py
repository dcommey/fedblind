import numpy as np
import os
import urllib.request
from scipy.io import loadmat
from .data_loader import create_dirichlet_distributed_data, preprocess_data
from utils.config import SVHN_DIR

SVHN_URL_TRAIN = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
SVHN_URL_TEST = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

def download_svhn():
    train_path = os.path.join(SVHN_DIR, 'train_32x32.mat')
    test_path = os.path.join(SVHN_DIR, 'test_32x32.mat')
    
    if not os.path.exists(train_path):
        print("Downloading SVHN train dataset...")
        urllib.request.urlretrieve(SVHN_URL_TRAIN, train_path)
    if not os.path.exists(test_path):
        print("Downloading SVHN test dataset...")
        urllib.request.urlretrieve(SVHN_URL_TEST, test_path)
    print("SVHN dataset downloaded.")

def load_svhn(num_clients, alpha):
    download_svhn()
    
    train_path = os.path.join(SVHN_DIR, 'train_32x32.mat')
    test_path = os.path.join(SVHN_DIR, 'test_32x32.mat')
    
    train_data = loadmat(train_path)
    test_data = loadmat(test_path)
    
    x_train = train_data['X']
    y_train = train_data['y'].flatten()
    x_test = test_data['X']
    y_test = test_data['y'].flatten()
    
    # SVHN labels are from 1-10, we need 0-9
    y_train -= 1
    y_test -= 1
    
    # Transpose to have channels_last format
    x_train = x_train.transpose((3, 0, 1, 2))
    x_test = x_test.transpose((3, 0, 1, 2))
    
    input_shape = (32, 32, 3)
    x_train = preprocess_data(x_train, input_shape)
    x_test = preprocess_data(x_test, input_shape)
    
    clients_data = create_dirichlet_distributed_data(x_train, y_train, num_clients, alpha)
    
    return clients_data, x_test, y_test