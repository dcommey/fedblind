# utils/config.py

import os

# Base directory for all datasets
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')

# Specific directories for each dataset
MNIST_DIR = os.path.join(DATA_DIR, 'mnist')
CIFAR10_DIR = os.path.join(DATA_DIR, 'cifar10')
CIFAR100_DIR = os.path.join(DATA_DIR, 'cifar100')
SVHN_DIR = os.path.join(DATA_DIR, 'svhn')
HAR_DIR = os.path.join(DATA_DIR, 'har')

# Create directories if they don't exist
for dir in [DATA_DIR, MNIST_DIR, CIFAR10_DIR, CIFAR100_DIR, SVHN_DIR, HAR_DIR]:
    os.makedirs(dir, exist_ok=True)

class Config:
    def __init__(self):
        """Initialize configuration with default values."""
        # Basic parameters
        self.algorithm = 'fedavg'  # one of ['fedavg', 'fedprox', 'scaffold', 'fednova']
        self.dataset_name = 'mnist'
        self.num_clients = 10
        self.num_rounds = 50
        self.batch_size = 32
        self.learning_rate = 0.01
        self.local_epochs = 5
        self.alpha = 0.5  # Dirichlet distribution parameter for non-iid data
        self.num_epochs = 1  # Number of local epochs
        
        # Clustering parameters
        self.use_clustering = False
        self.clustering = None 
        self.num_clusters = 2
        self.use_dp_clustering = False
        self.cl_epsilon = 1.0
        self.num_quantiles = 5
        self.cluster_on = 'features'  # one of ['features', 'labels']
        
        # Knowledge distillation parameters
        self.use_knowledge_distillation = False
        self.distillation_temperature = 2.0
        self.distillation_alpha = 0.5  # Weight for soft targets
        self.distillation_learning_rate = 0.001
        self.distillation_epochs = 50
        
        # Unlabeled data parameters
        self.unlabeled_ratio = 0.3
        self.labeled_subset_size = 1000
        
        # Tracking parameters
        self.target_accuracy = 0.8
        self.accuracy_thresholds = [0.6, 0.7, 0.8]
        
        # FedProx specific
        self.proximal_mu = 0.01
        
        # Model
        self.model_class = None
        self.TeacherModel = None

    def update(self, **kwargs):
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Config has no attribute '{key}'")

    def __str__(self):
        """Return a string representation of the configuration."""
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

# Create a default configuration
default_config = Config()