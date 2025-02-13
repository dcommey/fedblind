# FedSiKD: Federated Learning with Secure Clustering and Knowledge Distillation

FedSiKD is a framework that combines Federated Learning with secure (differentially private) client clustering and knowledge distillation to address data heterogeneity and client drift in federated learning systems while maintaining privacy guarantees.

## Key Features

- **Privacy-Preserving Client Clustering**: Uses differential privacy to securely group similar clients
- **Federated Learning**: Implements multiple FL algorithms (FedAvg, FedProx, Scaffold, FedNova)
- **Knowledge Distillation**: Transfers knowledge from multiple teacher models to a single student model
- **Quantile-Based Clustering**: Efficient clustering mechanism with privacy guarantees
- **Comprehensive Evaluation**: Includes metrics for accuracy, privacy, and convergence

## Project Structure

```
FedSiKD/
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
└── utils/
    └── config.py
```

## Usage

Run experiments with different configurations:

```bash
# Basic usage
python run_experiments.py --dataset cifar10 --num_clients 10 --num_rounds 50

# With clustering and privacy
python run_experiments.py --dataset cifar10 --num_clients 10 --num_rounds 50 \
    --num_clusters 2 --cl_epsilon 1.0 --alpha 0.5 --use_dp_clustering
```

### Command Line Arguments

- `--dataset`: Choose dataset (mnist, cifar10, cifar100, svhn, har)
- `--num_clients`: Number of federated clients
- `--num_rounds`: Number of communication rounds
- `--num_clusters`: Number of client clusters
- `--cl_epsilon`: Privacy budget for DP clustering
- `--alpha`: Dirichlet parameter for non-IID data distribution
- `--use_dp_clustering`: Enable differential privacy in clustering

## Supported Datasets

- MNIST
- CIFAR-10
- CIFAR-100
- SVHN
- HAR (Human Activity Recognition)

## Key Components

1. **Secure Clustering**
   - Differentially private quantile-based clustering
   - Feature-based and label-based clustering options
   - Privacy budget management

2. **Federated Learning**
   - Multiple federated optimization algorithms
   - Support for non-IID data distributions
   - Client-server architecture

3. **Knowledge Distillation**
   - Multi-teacher knowledge distillation
   - Student model optimization
   - Temperature scaling

## Results

The framework achieves:
- Improved model performance in heterogeneous settings
- Privacy guarantees through differential privacy
- Reduced communication overhead
- Better convergence compared to baseline methods

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{fedsikd2025,
  title={FedSiKD: Federated Learning with Secure Clustering and Knowledge Distillation},
  author={},
  journal={ArXiv},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

