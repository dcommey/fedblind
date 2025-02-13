import os
import json
import logging
from datetime import datetime
import numpy as np
import torch
from utils.config import Config
from federated.federated_framework import FederatedLearningFramework
from data.data_loader import get_dataloader
import argparse

def make_serializable(obj):
        """
        Recursively convert objects to be JSON serializable.
        Converts types (class objects) to strings.
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(item) for item in obj)
        elif isinstance(obj, type):
            return str(obj)
        else:
            return str(obj)

class ExperimentRunner:
    def __init__(self, args):
        self.setup_logging()
        self.args = args
        self.baselines = ['fedavg', 'fedprox', 'scaffold', 'fednova']
        self.alphas = [0.1, 0.5, 1.0]
        self.results_dir = self._create_results_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_results_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories for different result types
        for subdir in ['baselines', 'framework', 'summary']:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
            
        return results_dir

    def _get_data_loaders(self, config):
        """Initialize data loaders with current configuration"""
        client_dataloaders, test_loader, unlabeled_loader, labeled_subset_loader = get_dataloader(
            dataset_name=config.dataset_name,
            num_clients=config.num_clients,
            unlabeled_ratio=config.unlabeled_ratio,
            alpha=config.alpha,
            batch_size=config.batch_size,
            labeled_subset_size=config.labeled_subset_size
        )
        return client_dataloaders, test_loader, unlabeled_loader, labeled_subset_loader

    def save_results(self, results, category, experiment_name):
        import os, json
        # Build the full file path using os.path.join for cross-platform compatibility.
        filepath = os.path.join(self.results_dir, category, f"{experiment_name}.json")
        # Ensure the directory exists; if not, create it.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert results to a JSON-serializable format.
        serializable_results = make_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        self.logger.info(f"Saved results to {filepath}")

    def run_single_experiment(self, config, experiment_name):
        """Run a single experiment with proper initialization"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Configuration:")
        self.logger.info(f"- Algorithm: {config.algorithm}")
        self.logger.info(f"- Dataset: {config.dataset_name}")
        self.logger.info(f"- Alpha (non-IID degree): {config.alpha}")
        self.logger.info(f"- Number of rounds: {config.num_rounds}")
        if config.use_clustering:
            self.logger.info(f"- Clustering: {'DP' if self.args.use_dp_clustering else 'Non-DP'}")
            self.logger.info(f"- Number of clusters: {config.num_clusters}")
        if config.use_knowledge_distillation:
            self.logger.info(f"- Using Knowledge Distillation")
        self.logger.info(f"{'='*50}\n")
        
        # Get data loaders
        client_dataloaders, test_loader, unlabeled_loader, labeled_subset_loader = self._get_data_loaders(config)
        
        # Initialize framework
        framework = FederatedLearningFramework(config)
        framework.initialize(
            client_dataloaders=client_dataloaders,
            test_loader=test_loader, 
            unlabeled_loader=unlabeled_loader,
            labeled_subset_loader=labeled_subset_loader
        )
        
        # Train and collect results
        self.logger.info("Starting training...")
        current_round = 0
        
        def progress_callback(round_num, metrics):
            nonlocal current_round
            if round_num > current_round:
                current_round = round_num
                accuracy = metrics.get('accuracy', 'N/A')
                loss = metrics.get('loss', 'N/A')
                
                # Format accuracy and loss only if they are numbers
                accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else accuracy
                loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else loss
                
                self.logger.info(f"Round {round_num}/{config.num_rounds} - "
                            f"Accuracy: {accuracy_str}, "
                            f"Loss: {loss_str}")
        
        # Add progress_callback to framework training
        framework.train(progress_callback=progress_callback)
        
        self.logger.info(f"\nExperiment {experiment_name} completed.")
        if config.use_knowledge_distillation:
            final_results = framework.get_student_model_results()
            if final_results:
                accuracy = final_results.get('accuracy', 'N/A')
                loss = final_results.get('loss', 'N/A')
                f1 = final_results.get('f1_score', 'N/A')
                
                # Format metrics only if they are numbers
                accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else str(accuracy)
                loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else str(loss)
                f1_str = f"{f1:.4f}" if isinstance(f1, (float, int)) else str(f1)
                
                self.logger.info(f"Final Student Model Results:")
                self.logger.info(f"- Accuracy: {accuracy_str}")
                self.logger.info(f"- Loss: {loss_str}")
                self.logger.info(f"- F1 Score: {f1_str}")
        else:
            # For baselines, perform a final evaluation on the global model
            # Prepare test data as in the training loop
            test_data = torch.cat([batch[0] for batch in framework.test_loader], dim=0)
            test_labels = torch.cat([batch[1] for batch in framework.test_loader], dim=0)
            final_results = framework.server.evaluate_global_model(test_data, test_labels)

        results = {
            'accuracy_history': framework.get_accuracy_history(),
            'loss_history': framework.get_loss_history(),
            'cluster_assignments': framework.get_cluster_assignments() if config.use_clustering else None,
            'rounds_to_target': framework.rounds_to_target,
            'final_results': final_results,
            'config': vars(config)
        }
        
        # Add privacy budget if using DP
        if config.use_dp_clustering:
            results['privacy_info'] = framework.get_privacy_info()
            
        # Add clustering info if using clustering
        if config.use_clustering:
            results['clustering_info'] = framework.get_clustering_info()
                
        return results

    def run_baselines(self):
        """Run baseline experiments."""
        baseline_results = {}
        alpha = self.args.alpha  # non-IID degree
        for baseline in self.baselines:
            baseline_results[baseline] = {}
            experiment_name = f"{baseline}_alpha_{alpha}"
            self.logger.info(f"Running baseline experiment: {experiment_name}")

            # Create and update config for each experiment
            config = Config()
            config.algorithm = baseline
            config.alpha = alpha
            config.dataset_name = self.args.dataset
            config.num_clients = self.args.num_clients
            config.num_rounds = self.args.num_rounds
            config.batch_size = self.args.batch_size
            config.learning_rate = self.args.learning_rate
            
            # IMPORTANT: Baselines should not use clustering or KD
            config.use_clustering = False
            config.use_knowledge_distillation = False
            config.num_clusters = 1  # No clustering for baselines

            # Set model based on dataset
            if config.dataset_name == 'mnist':
                from models.mnist_models import MNISTTeacher
                config.model_class = MNISTTeacher
                config.TeacherModel = MNISTTeacher
            elif config.dataset_name == 'cifar10':
                from models.cifar_models import CIFAR10Teacher
                config.model_class = CIFAR10Teacher
                config.TeacherModel = CIFAR10Teacher
            elif config.dataset_name == 'cifar100':
                from models.cifar_models import CIFAR100Teacher
                config.model_class = CIFAR100Teacher
                config.TeacherModel = CIFAR100Teacher
            elif config.dataset_name == 'svhn':
                from models.svhn_models import SVHNTeacher
                config.model_class = SVHNTeacher
                config.TeacherModel = SVHNTeacher
            elif config.dataset_name == 'har':
                from models.har_models import HARTeacher
                config.model_class = HARTeacher
                config.TeacherModel = HARTeacher

            # Now run the single experiment
            results = self.run_single_experiment(config, experiment_name)
            baseline_results[baseline][f"alpha_{alpha}"] = results
            
            # Save the results
            self.save_results(results, 'baseline', experiment_name)
            
        return baseline_results  # Moved outside the loop

    def run_framework(self):
        """Run framework experiments (baseline + clustering + KD)"""
        self.logger.info("Running framework experiments...")
        framework_results = {}
        
        alpha = self.args.alpha  # Use the alpha from args
        for baseline in self.baselines:
            framework_results[baseline] = {}
            experiment_name = f"{baseline}_framework_alpha_{alpha}"
            self.logger.info(f"Running framework: {experiment_name}")
            
            # Configure framework
            config = Config()
            config.algorithm = baseline
            config.alpha = alpha
            config.use_clustering = True
            config.use_knowledge_distillation = True
            config.dataset_name = self.args.dataset
            config.num_clients = self.args.num_clients
            config.num_rounds = self.args.num_rounds
            config.batch_size = self.args.batch_size
            config.learning_rate = self.args.learning_rate
            config.num_clusters = self.args.num_clusters
            config.cl_epsilon = self.args.cl_epsilon
            config.num_quantiles = self.args.num_quantiles
            config.cluster_on = self.args.cluster_on

            # Set model based on dataset
            if config.dataset_name == 'mnist':
                from models.mnist_models import MNISTTeacher
                config.model_class = MNISTTeacher
                config.TeacherModel = MNISTTeacher
            elif config.dataset_name == 'cifar10':
                from models.cifar_models import CIFAR10Teacher
                config.model_class = CIFAR10Teacher
                config.TeacherModel = CIFAR10Teacher
            elif config.dataset_name == 'cifar100':
                from models.cifar_models import CIFAR100Teacher
                config.model_class = CIFAR100Teacher
                config.TeacherModel = CIFAR100Teacher
            elif config.dataset_name == 'svhn':
                from models.svhn_models import SVHNTeacher
                config.model_class = SVHNTeacher
                config.TeacherModel = SVHNTeacher
            elif config.dataset_name == 'har':
                from models.har_models import HARTeacher
                config.model_class = HARTeacher
                config.TeacherModel = HARTeacher

            # Set clustering object based on DP flag
            if self.args.use_dp_clustering:
                from clustering.dp_quantile_clustering import DPQuantileClustering
                config.clustering = DPQuantileClustering(
                    max_clusters=config.num_clusters,
                    epsilon=config.cl_epsilon,
                    num_quantiles=config.num_quantiles,
                    dataset_name=config.dataset_name,
                    cluster_on=config.cluster_on
                )
            else:
                from clustering.non_dp_quantile_clustering import NonDPQuantileClustering
                config.clustering = NonDPQuantileClustering(
                    max_clusters=config.num_clusters,
                    num_quantiles=config.num_quantiles,
                    dataset_name=config.dataset_name,
                    cluster_on=config.cluster_on
                )

            # Run experiment
            results = self.run_single_experiment(config, experiment_name)
            framework_results[baseline][f"alpha_{alpha}"] = results
            
            # Save individual experiment results
            self.save_results(results, 'framework', experiment_name)
            
        return framework_results

    def generate_summary(self, baseline_results, framework_results):
        """Generate and save summary with comprehensive error handling."""
        try:
            if not baseline_results or not framework_results:
                self.logger.error("Missing results for summary generation")
                return {
                    "error": "Incomplete results",
                    "baseline_results_present": bool(baseline_results),
                    "framework_results_present": bool(framework_results)
                }

            baseline_summary = self._summarize_results(baseline_results)
            framework_summary = self._summarize_results(framework_results)
            improvements = self._calculate_improvements(baseline_results, framework_results)

            summary = {
                "baseline_summary": baseline_summary,
                "framework_summary": framework_summary,
                "improvements": improvements,
                "experiment_metadata": {
                    "total_experiments": len(self.baselines) * 2,
                    "dataset": self.args.dataset,
                    "num_clients": self.args.num_clients,
                    "num_rounds": self.args.num_rounds
                }
            }
            
            # Save even if some parts are empty
            self.save_results(summary, 'summary', 'experiment_summary')
            return summary
            
        except Exception as e:
            error_summary = {
                "error": str(e),
                "baseline_results_present": bool(baseline_results),
                "framework_results_present": bool(framework_results)
            }
            self.logger.error(f"Error generating summary: {str(e)}")
            return error_summary

    def _summarize_results(self, results):
        """Summarize results with proper error handling."""
        if not results:
            self.logger.warning("No results to summarize")
            return {}

        summary = {}
        try:
            for baseline in results:
                summary[baseline] = {}
                for alpha_key in results[baseline]:
                    exp_results = results[baseline][alpha_key]
                    if not exp_results:
                        self.logger.warning(f"No results for {baseline} {alpha_key}")
                        continue

                    final_results = exp_results.get("final_results", {}) or {}
                    privacy_info = exp_results.get("privacy_info", {}) or {}
                    clustering_info = exp_results.get("clustering_info", {}) or {}
                    
                    summary[baseline][alpha_key] = {
                        "final_accuracy": final_results.get("accuracy"),
                        "rounds_to_target": exp_results.get("rounds_to_target"),
                        "privacy_budget": privacy_info.get("privacy_budget_spent"),
                        "num_clusters": clustering_info.get("num_clusters")
                    }
        except Exception as e:
            self.logger.error(f"Error in _summarize_results: {str(e)}")
            return {}
        return summary

    def _calculate_improvements(self, baseline_results, framework_results):
        """Calculate improvements with proper error handling."""
        improvements = {}
        try:
            for baseline in self.baselines:
                if baseline not in baseline_results or baseline not in framework_results:
                    continue
                    
                improvements[baseline] = {}
                for alpha_key in baseline_results[baseline]:
                    if alpha_key not in framework_results[baseline]:
                        continue
                        
                    base_results = baseline_results[baseline][alpha_key].get("final_results", {}) or {}
                    framework_res = framework_results[baseline][alpha_key].get("final_results", {}) or {}
                    
                    base_acc = base_results.get("accuracy", 0)
                    framework_acc = framework_res.get("accuracy", 0)
                    
                    if base_acc > 0:
                        acc_improvement = ((framework_acc - base_acc) / base_acc) * 100
                    else:
                        acc_improvement = 0
                    
                    improvements[baseline][alpha_key] = {
                        "accuracy_improvement_percentage": acc_improvement,
                        "base_accuracy": base_acc,
                        "framework_accuracy": framework_acc
                    }
        except Exception as e:
            self.logger.error(f"Error in _calculate_improvements: {str(e)}")
            return {}
            
        return improvements

def parse_args():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["mnist", "cifar10", "cifar100", "svhn", "har"],
                        help="Dataset to use")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=50,
                        help="Number of communication rounds")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--num_clusters", type=int, default=2,
                        help="Number of clusters for the framework")
    parser.add_argument("--cl_epsilon", type=float, default=1.0,
                        help="Epsilon for DP clustering")
    parser.add_argument("--use_dp_clustering", action='store_true',
                        help="Use Differentially Private (DP) clustering")
    parser.add_argument("--num_quantiles", type=int, default=5,
                        help="Number of quantiles for clustering")
    parser.add_argument("--cluster_on", choices=['features', 'labels'], default='features', 
                        help="Cluster on features or labels")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet distribution parameter for non-iid data generation")
    return parser.parse_args()

def main():
    args = parse_args()
    runner = ExperimentRunner(args)
    
    try:
        # Calculate total experiments
        num_baselines = len(runner.baselines)
        total_experiments = (
            num_baselines +  # Baseline experiments
            num_baselines   # Framework experiments
        )
        current_experiment = 0
        
        def update_progress():
            nonlocal current_experiment
            current_experiment += 1
            progress = (current_experiment / total_experiments) * 100
            runner.logger.info(f"\nOverall Progress: [{current_experiment}/{total_experiments}] "
                             f"({progress:.1f}%)\n")
        
        # Run baselines
        runner.logger.info("\n" + "="*70)
        runner.logger.info("Starting Baseline Experiments")
        runner.logger.info("="*70)
        baseline_results = runner.run_baselines()
        update_progress()
        
        # Run framework
        runner.logger.info("\n" + "="*70)
        runner.logger.info("Starting Framework Experiments (Baseline + Clustering + KD)")
        runner.logger.info("="*70)
        framework_results = runner.run_framework()
        update_progress()
        
        # Generate and save summary
        summary = runner.generate_summary(baseline_results, framework_results)
        
        # Print final summary
        runner.logger.info("\n" + "="*70)
        runner.logger.info("Experiment Summary:")
        runner.logger.info("="*70)
        runner.logger.info(json.dumps(summary, indent=2))
        
    except Exception as e:
        runner.logger.error(f"Error during experiment: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()