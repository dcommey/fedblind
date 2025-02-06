from tensorflow import keras
from .data_loader import create_dirichlet_distributed_data, preprocess_data
from utils.config import MNIST_DIR

def load_mnist(num_clients, alpha):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=f'{MNIST_DIR}/mnist.npz')
    
    input_shape = (28, 28, 1)
    x_train = preprocess_data(x_train, input_shape)
    x_test = preprocess_data(x_test, input_shape)
    
    clients_data = create_dirichlet_distributed_data(x_train, y_train, num_clients, alpha)
    
    return clients_data, x_test, y_test