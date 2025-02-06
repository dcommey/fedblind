import argparse
import logging
import numpy as np
import torch
from run_baseline import set_random_seed, run_experiment
import matplotlib.pyplot as plt
from datetime import datetime
import json

def test_baselines(dataset='mnist', num_clients=10, num_rounds=5, local_epochs=2, batch_size=32):
    baselines = ['fedavg', 'fedprox', 'scaffold', 'fednova']
    results = {}
    
    # Common arguments
    class Args:
        def __init__(self):
            self.dataset = dataset
            self.num_clients = num_clients
            self.num_rounds = num_rounds
            self.local_epochs = local_epochs
            self.batch_size = batch_size
            self.proximal_mu = 0.01
            self.seed = 42
            self.learning_rate = 0.01
            self.alpha = 0.5
            self.unlabeled_ratio = 0.3
            self.target_accuracy = None
            self.use_dp_clustering = False
            self.max_clusters = 1
            self.temperature = 2.0
            self.distillation_epochs = 10
            self.distillation_learning_rate = 0.001
            self.cluster_on = 'features'
            self.use_dp_fl = False
            self.fl_epsilon = 5.0
            self.cl_epsilon = 5.0
            self.fl_delta = 1e-5
            self.max_grad_norm = 1.0
            self.num_quantiles = 5

    # Test each baseline
    for baseline in baselines:
        print(f"\nTesting {baseline}...")
        args = Args()
        args.baseline = baseline
        set_random_seed(args.seed)
        
        try:
            result = run_experiment(args)
            if result is not None:
                results[baseline] = result
                print(f"Completed {baseline} with final result: {result}")
            else:
                print(f"No results returned for {baseline}")
        except Exception as e:
            print(f"Error running {baseline}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[baseline] = None

    # Plot results if we have any
    if any(results.values()):
        plt.figure(figsize=(12, 8))
        
        for baseline, result in results.items():
            if result is not None and 'accuracy_history' in result:
                plt.plot(result['accuracy_history'], label=baseline)
        
        plt.title(f'Baseline Comparison on {dataset.upper()} Dataset')
        plt.xlabel('Rounds')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'baseline_comparison_{dataset}_{timestamp}.png')
        
        # Save results
        with open(f'baseline_results_{dataset}_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all baseline algorithms")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "svhn", "har"], 
                    default="mnist", help="Dataset to use")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    results = test_baselines(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size
    ) 