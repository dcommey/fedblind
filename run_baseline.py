import argparse
import logging
import json
import random
import numpy as np
import torch
from datetime import datetime
from utils.config import Config
from federated.federated_framework import FederatedLearningFramework
from data.data_loader import get_dataloader
from clustering.non_dp_quantile_clustering import NonDPQuantileClustering
from clustering.dp_quantile_clustering import DPQuantileClustering

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Federated Learning Experiments")
    parser.add_argument("--baseline", choices=["fedavg", "fedprox", "scaffold", "fednova"], 
                        required=True, help="Baseline algorithm to run")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "svhn", "har"], 
                        required=True, help="Dataset to use")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--proximal_mu", type=float, default=0.01, help="FedProx mu parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter")
    parser.add_argument("--unlabeled_ratio", type=float, default=0.3, help="Unlabeled ratio")
    parser.add_argument("--target_accuracy", type=float, help="Target accuracy")
    parser.add_argument("--use_dp_clustering", action='store_true', help="Use DP clustering")
    parser.add_argument("--max_clusters", type=int, default=1, help="Max clusters")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature")
    parser.add_argument("--distillation_epochs", type=int, default=10, help="Distillation epochs")
    parser.add_argument("--distillation_learning_rate", type=float, default=0.001, help="Distillation learning rate")
    parser.add_argument("--cluster_on", choices=['features', 'labels'], default='features', 
                        help="Cluster on features or labels")
    parser.add_argument("--use_dp_fl", action='store_true', help="Use DP FL")
    parser.add_argument("--fl_epsilon", type=float, default=5.0, help="FL epsilon")
    parser.add_argument("--cl_epsilon", type=float, default=5.0, help="DP clustering epsilon")
    parser.add_argument("--fl_delta", type=float, default=1e-5, help="FL delta")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max grad norm")
    parser.add_argument("--num_quantiles", type=int, default=5, help="Number of quantiles")
    return parser.parse_args()

def run_experiment(args):
    # Build config
    config = Config()
    config.dataset_name = args.dataset
    config.num_clients = args.num_clients
    config.num_rounds = args.num_rounds
    config.local_epochs = args.local_epochs
    config.batch_size = args.batch_size
    config.algorithm = args.baseline
    config.proximal_mu = args.proximal_mu
    config.learning_rate = args.learning_rate
    config.alpha = args.alpha
    config.unlabeled_ratio = args.unlabeled_ratio
    config.target_accuracy = args.target_accuracy
    config.use_dp_clustering = args.use_dp_clustering
    config.max_clusters = args.max_clusters
    config.distillation_temperature = args.temperature
    config.distillation_epochs = args.distillation_epochs
    config.distillation_learning_rate = args.distillation_learning_rate

    # Set model class based on dataset
    if args.dataset == 'mnist':
        from models.mnist_models import MNISTTeacher
        config.model_class = MNISTTeacher
        config.TeacherModel = MNISTTeacher
    elif args.dataset == 'cifar10':
        from models.cifar_models import CIFAR10Teacher
        config.model_class = CIFAR10Teacher
        config.TeacherModel = CIFAR10Teacher
    elif args.dataset == 'cifar100':
        from models.cifar_models import CIFAR100Teacher
        config.model_class = CIFAR100Teacher
        config.TeacherModel = CIFAR100Teacher
    elif args.dataset == 'svhn':
        from models.svhn_models import SVHNTeacher
        config.model_class = SVHNTeacher
        config.TeacherModel = SVHNTeacher
    elif args.dataset == 'har':
        from models.har_models import HARTeacher
        config.model_class = HARTeacher
        config.TeacherModel = HARTeacher
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Assign clustering (DP or non-DP)
    if args.use_dp_clustering:
        config.clustering = DPQuantileClustering(epsilon=args.cl_epsilon, num_quantiles=args.num_quantiles)
    else:
        config.clustering = NonDPQuantileClustering(num_quantiles=args.num_quantiles)
    config.use_clustering = (args.max_clusters > 1) 

    # Load data with labeled_subset_size
    client_dataloaders, test_loader, unlabeled_loader, labeled_subset_loader = get_dataloader(
        config.dataset_name, 
        config.num_clients, 
        config.unlabeled_ratio,
        config.alpha,
        config.batch_size,
        labeled_subset_size=config.labeled_subset_size
    )

    # Build the federated framework
    framework = FederatedLearningFramework(config)
    
    # Initialize with all loaders
    framework.initialize(
        client_dataloaders=client_dataloaders,
        test_loader=test_loader,
        unlabeled_loader=unlabeled_loader,
        labeled_subset_loader=labeled_subset_loader
    )
    
    # Train the framework
    framework.train()
    
    return framework.get_student_model_results()

def main():
    args = parse_arguments()
    set_random_seed(args.seed)
    run_experiment(args)

if __name__ == "__main__":
    main()
