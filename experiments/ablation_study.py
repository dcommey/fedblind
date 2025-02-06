import json
import os
import time
from ..utils.config import Config
from ..utils.metrics import evaluate_federated_learning, print_federated_metrics
from ..data.data_loader import load_and_preprocess_data
from ..algorithms.fedsiKD import FedSiKD

def run_ablation_study(base_config):
    # Load data
    clients_data, (x_test, y_test) = load_and_preprocess_data(
        dataset=base_config.dataset,
        num_clients=base_config.num_clients,
        alpha=base_config.dirichlet_alpha,
        augment=base_config.use_augmentation
    )

    # Define ablation configurations
    ablation_configs = [
        ('Full FedSiKD', base_config),
        ('No Clustering', disable_clustering(base_config)),
        ('No Knowledge Distillation', disable_knowledge_distillation(base_config)),
        ('No Differential Privacy', disable_differential_privacy(base_config)),
        ('Only FedAvg', disable_all_components(base_config))
    ]

    results = {}
    for name, config in ablation_configs:
        fedsiKD = FedSiKD(config)
        start_time = time.time()
        model = fedsiKD.run(clients_data)
        training_time = time.time() - start_time

        privacy_params = fedsiKD.get_privacy_params()
        client_features = fedsiKD.get_client_features()
        cluster_assignments = fedsiKD.get_cluster_assignments()

        metrics = evaluate_federated_learning(
            config=config,
            model=model,
            x_test=x_test,
            y_test=y_test,
            privacy_params=privacy_params,
            training_time=training_time,
            client_features=client_features,
            cluster_assignments=cluster_assignments
        )

        results[name] = metrics

        print(f"{name} Results:")
        print_federated_metrics(metrics)

    # Save results
    save_results(results, base_config)

    return results

def disable_clustering(config):
    new_config = Config()
    new_config.__dict__.update(config.__dict__)
    new_config.use_clustering = False
    return new_config

def disable_knowledge_distillation(config):
    new_config = Config()
    new_config.__dict__.update(config.__dict__)
    new_config.use_knowledge_distillation = False
    return new_config

def disable_differential_privacy(config):
    new_config = Config()
    new_config.__dict__.update(config.__dict__)
    new_config.use_differential_privacy = False
    return new_config

def disable_all_components(config):
    new_config = Config()
    new_config.__dict__.update(config.__dict__)
    new_config.use_clustering = False
    new_config.use_knowledge_distillation = False
    new_config.use_differential_privacy = False
    return new_config

def save_results(results, config):
    os.makedirs('results', exist_ok=True)
    filename = (
        f"ablation_study_{config.dataset}_"
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
    base_config = Config()
    run_ablation_study(base_config)