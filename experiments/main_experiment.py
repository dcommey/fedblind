import json
import os
import time
from ..utils.config import Config
from ..utils.metrics import evaluate_federated_learning, print_federated_metrics
from ..data.data_loader import load_and_preprocess_data
from ..algorithms.fedsiKD import FedSiKD
from ..baselines.fedavg import FedAvg
from ..baselines.fedprox import FedProx
from ..baselines.fl_hc import FLHC

def run_experiment(config):
    # Load data
    clients_data, (x_test, y_test) = load_and_preprocess_data(
        dataset=config.dataset,
        num_clients=config.num_clients,
        alpha=config.dirichlet_alpha,
        augment=config.use_augmentation
    )

    # Initialize and run FedSiKD
    fedsiKD = FedSiKD(config)
    start_time = time.time()
    fedsiKD_model = fedsiKD.run(clients_data)
    fedsiKD_time = time.time() - start_time

    # Get additional information for metrics
    privacy_params = fedsiKD.get_privacy_params()
    client_features = fedsiKD.get_client_features()
    cluster_assignments = fedsiKD.get_cluster_assignments()

    # Evaluate FedSiKD
    fedsiKD_metrics = evaluate_federated_learning(
        config=config,
        model=fedsiKD_model,
        x_test=x_test,
        y_test=y_test,
        privacy_params=privacy_params,
        training_time=fedsiKD_time,
        client_features=client_features,
        cluster_assignments=cluster_assignments
    )

    print("FedSiKD Results:")
    print_federated_metrics(fedsiKD_metrics)

    # Run baselines
    baselines = [
        ('FedAvg', FedAvg(config)),
        ('FedProx', FedProx(config)),
        ('FLHC', FLHC(config))
    ]

    baseline_results = {}
    for name, baseline in baselines:
        start_time = time.time()
        baseline_model = baseline.run(clients_data)
        baseline_time = time.time() - start_time

        baseline_metrics = evaluate_federated_learning(
            config=config,
            model=baseline_model,
            x_test=x_test,
            y_test=y_test,
            privacy_params=None,
            training_time=baseline_time,
            client_features=None,
            cluster_assignments=None
        )

        baseline_results[name] = baseline_metrics

        print(f"{name} Results:")
        print_federated_metrics(baseline_metrics)

    results = {
        'FedSiKD': fedsiKD_metrics,
        'FedAvg': baseline_results['FedAvg'],
        'FedProx': baseline_results['FedProx'],
        'FLHC': baseline_results['FLHC']
    }

    # Save results
    save_results(results, config)

    return results

def save_results(results, config):
    os.makedirs('results', exist_ok=True)
    filename = (
        f"main_experiment_{config.dataset}_"
        f"C{config.num_clients}_"
        f"R{config.num_rounds}_"
        f"E{config.local_epochs}_"
        f"B{config.batch_size}_"
        f"LR{config.learning_rate}_"
        f"A{config.dirichlet_alpha}_"
        f"CL{'T' if config.use_clustering else 'F'}_"
        f"KD{'T' if config.use_knowledge_distillation else 'F'}_"
        f"DP{'T' if config.use_differential_privacy else 'F'}"
    )
    if config.use_differential_privacy:
        filename += f"_DPe{config.dp_epsilon}_DPd{config.dp_delta}_DPc{config.dp_clip_norm}"
    
    filename += ".json"

    with open(os.path.join('results', filename), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    config = Config()
    run_experiment(config)