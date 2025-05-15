# FedBlind: Federated Learning with Secure Clustering and Knowledge Distillation

## Overview

FedBlind is a novel framework that integrates Federated Learning (FL), differentially private (DP) client clustering, and knowledge distillation to address data heterogeneity and client drift in federated systems while maintaining rigorous privacy guarantees.

## Key Features

- **Privacy-Preserving Client Clustering**: Implements DP quantile-based clustering to securely group clients based on feature or label similarity.
- **Federated Learning Algorithms**: Supports multiple FL optimization strategies, including FedAvg, FedProx, SCAFFOLD, and FedNova.
- **Knowledge Distillation**: Applies multi-teacher to single-student distillation to aggregate knowledge across clusters.
- **Quantile-Based Clustering**: Efficient and privacy-aware method for clustering non-IID clients.
- **Evaluation Suite**: Benchmarks include accuracy, convergence speed, and privacy utility trade-offs.

## Project Structure

```
FedBlind/
├── baselines/
│   ├── fedavg.py
│   ├── fedprox.py
│   ├── scaffold.py
│   └── fednova.py
├── clustering/
│   ├── dp_quantile_clustering.py
│   └── non_dp_quantile_clustering.py
├── data/
│   └── data_loader.py
├── federated/
│   ├── federated_framework.py
│   ├── base_server.py
│   └── base_client.py
├── knowledge_distillation/
│   ├── distiller.py
│   └── teacher_student_models.py
├── models/
│   ├── mnist_models.py
│   ├── cifar_models.py
│   └── har_models.py
├── utils/
├── config.py
└── run_experiments.py
```

## Usage

```bash
# Basic experiment
python run_experiments.py --dataset har --num_clients 10 --num_rounds 50

# With DP clustering
python run_experiments.py --dataset har --num_clients 10 --num_rounds 50 \
    --num_clusters 2 --cl_epsilon 1.0 --alpha 0.5 --use_dp_clustering
```

### Command Line Arguments

- `--dataset`: Dataset to use (`mnist`, `cifar10`, `cifar100`, `svhn`, `har`)
- `--num_clients`: Number of clients
- `--num_rounds`: Number of communication rounds
- `--num_clusters`: Number of clusters for client grouping
- `--cl_epsilon`: DP budget for clustering
- `--alpha`: Dirichlet parameter for data heterogeneity
- `--use_dp_clustering`: Enables differentially private clustering

## Supported Datasets

- MNIST
- CIFAR-10
- CIFAR-100
- SVHN
- HAR (Human Activity Recognition)

## Methodology Summary

### Secure Clustering

- Differentially private, quantile-based mechanism
- Label-based or feature-based grouping
- Controlled via privacy budget `epsilon`

### Federated Learning

- Modular optimization algorithms: FedAvg, FedProx, SCAFFOLD, FedNova
- Handles non-IID client data via Dirichlet-sampled partitions

### Knowledge Distillation

- Multi-teacher ensemble from each cluster
- Temperature-scaled soft target transfer to a student model
- Improves generalization without direct data sharing

## Results

FedBlind demonstrates:

- Improved convergence and final accuracy in heterogeneous FL settings
- Strong privacy guarantees using DP clustering
- Reduction in communication cost
- Better performance than non-clustered and non-distilled baselines

## Citation

This repository accompanies a submission under double-blind review. To refer to this work:

```bibtex
@unpublished{anonymous2025fedblind,
  title={{FedBlind}: Federated Learning with Secure Clustering and Knowledge Distillation},
  author={Anonymous Authors},
  note={Under review at NeurIPS 2025},
  year={2025}
}
```

## License

This project is released under the MIT License.
