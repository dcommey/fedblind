from tensorflow import keras
from .data_loader import create_dirichlet_distributed_data, preprocess_data
from utils.config import CIFAR10_DIR

def load_cifar10(num_clients, alpha):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    input_shape = (32, 32, 3)
    x_train = preprocess_data(x_train, input_shape)
    x_test = preprocess_data(x_test, input_shape)
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    clients_data = create_dirichlet_distributed_data(x_train, y_train, num_clients, alpha)
    
    return clients_data, x_test, y_test