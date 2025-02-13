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
        # Dataset configuration
        self.dataset_name = 'cifar10'  # Options: 'mnist', 'cifar10', 'cifar100', 'svhn', 'har'
        self.num_clients = 100
        self.alpha = 0.5  # Dirichlet distribution parameter
        self.unlabeled_ratio = 0.3
        self.use_augmentation = False

        # Federated learning configuration
        self.num_rounds = 100
        self.local_epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.001
        self.target_accuracy = None

        # Clustering configuration
        self.use_clustering = False
        self.num_clusters = 10
        self.cluster_on = 'features'  # Options: 'features', 'labels'
        self.num_quantiles = 5
        self.clustering = None 

        # Distillation configuration
        self.use_knowledge_distillation = True
        self.distillation_alpha = 0.5
        self.distillation_temperature = 2.0
        self.distillation_epochs = 50
        self.distillation_learning_rate = 0.001
        self.labeled_subset_size = 1000

        # Differential privacy configuration
        self.use_differential_privacy = False  # Set to False by default
        self.use_dp_model_training = False
        self.use_dp_clustering = False
        self.epsilon_1 = 0.1  # For clustering
        self.epsilon_2 = 0.5  # For model training
        self.delta = 1e-5
        self.dp_l2_norm_clip = 1.0
        self.dp_noise_multiplier = 1.1
        self.dp_num_microbatches = 1

        # Algorithm-specific parameters
        self.algorithm = 'fedavg'  # Default algorithm
        self.proximal_mu = 0.01    # FedProx hyperparameter

        # Model related
        self.model_class = None    # Will be set based on dataset
        self.TeacherModel = None   # Will be set based on dataset

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