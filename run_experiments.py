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
import random
from scipy import stats
import time

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
        self.set_seeds(42)
        self.setup_logging()
        self.args = args
        self.baselines = ['fedavg', 'fedprox', 'scaffold', 'fednova']
        self.alphas = [0.1, 0.5, 1.0]
        self.results_dir = self._get_results_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_seeds(self, seed):
        """Set seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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

    def _get_results_dir(self):
        """Get or create results directory with proper organization."""
        # Check if results directory exists, otherwise create it
        base_results_dir = "results"
        
        if not os.path.exists(base_results_dir):
            os.makedirs(base_results_dir, exist_ok=True)
            
        # Create subdirectories for different result types if they don't exist
        for subdir in ['baselines', 'framework', 'summary']:
            subdir_path = os.path.join(base_results_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)
        
        # Add timestamp only to log file, not to the directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add run info to log
        self.logger.info(f"Experiment run started at {timestamp}")
        self.logger.info(f"Saving results to directory: {base_results_dir}")
        
        return base_results_dir

    def _save_experiment_results(self, results, experiment_name):
        """Save experiment results using consistent directory structure."""
        # Determine appropriate subdirectory based on experiment name
        if '_baseline_' in experiment_name:
            subdir = 'baselines'
        elif '_framework_' in experiment_name:
            subdir = 'framework'
        else:
            subdir = 'other'
        
        # Build the file path
        filepath = os.path.join(self.results_dir, subdir, f"{experiment_name}.json")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Make results JSON serializable
        serializable_results = self._make_serializable(results)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        self.logger.info(f"Saved results to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, type):
            return str(obj)
        else:
            return str(obj)

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
        """Save results to appropriate subdirectory."""
        filepath = os.path.join(self.results_dir, category, f"{experiment_name}.json")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert results to a JSON-serializable format
        serializable_results = self._make_serializable(results)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        self.logger.info(f"Saved results to {filepath}")

    def run_single_experiment(self, config, experiment_name):
        """Run a single experiment with the given configuration."""
        try:
            start_time = time.time()
            
            # Set up clustering if needed
            if getattr(config, 'use_dp_clustering', False):
                from clustering.dp_quantile_clustering import DPQuantileClustering
                config.clustering = DPQuantileClustering(
                    max_clusters=config.num_clusters,
                    epsilon=config.cl_epsilon,
                    num_quantiles=config.num_quantiles,
                    dataset_name=config.dataset_name,
                    cluster_on=config.cluster_on
                )
                config.use_clustering = True
            elif getattr(config, 'num_clusters', 1) > 1:
                from clustering.non_dp_quantile_clustering import NonDPQuantileClustering
                config.clustering = NonDPQuantileClustering(
                    max_clusters=config.num_clusters,
                    num_quantiles=config.num_quantiles,
                    dataset_name=config.dataset_name,
                    cluster_on=config.cluster_on
                )
                config.use_clustering = True
            else:
                config.clustering = None
                config.use_clustering = False
            
            # Initialize the framework
            framework = FederatedLearningFramework(config)
            
            # Load data
            client_dataloaders, test_loader, unlabeled_loader, labeled_subset_loader = get_dataloader(
                config.dataset_name, 
                config.num_clients,
                config.unlabeled_ratio,
                config.alpha,
                config.batch_size,
                labeled_subset_size=config.labeled_subset_size
            )
            
            # Initialize framework with data
            framework.initialize(
                client_dataloaders=client_dataloaders,
                test_loader=test_loader,
                unlabeled_loader=unlabeled_loader,
                labeled_subset_loader=labeled_subset_loader
            )
            
            # Run training and capture results
            training_results = framework.train()
            
            # Ensure we have a dictionary for results
            if training_results is None:
                training_results = {}
            
            # Add experiment metadata
            results = {
                'experiment_name': experiment_name,
                'runtime': time.time() - start_time,
                'config': self._get_serializable_config(config),
                'final_results': {
                    'accuracy': training_results.get('final_accuracy'),
                    'loss': training_results.get('final_loss'),
                    'f1_score': training_results.get('final_f1_score')
                },
                'accuracy_history': self._extract_accuracy_history(training_results),
                'loss_history': self._extract_loss_history(training_results),
                'clustering_info': {
                    'num_clusters': training_results.get('num_clusters', 1),
                    'cluster_sizes': training_results.get('cluster_sizes', [config.num_clients])
                },
                'rounds_to_target': training_results.get('rounds_to_target')
            }
            
            # Save results
            self.save_results(results, 'baseline' if 'framework' not in experiment_name else 'framework', experiment_name)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error during experiment: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _extract_accuracy_history(self, results):
        """Extract accuracy history from results in a consistent format."""
        # Check for distillation results first
        if 'distillation_results' in results and 'val_accuracy_history' in results['distillation_results']:
            return results['distillation_results']['val_accuracy_history']
            
        # Check for cluster accuracy history
        if 'accuracy_history' in results:
            # If it's a dict with cluster IDs as keys, flatten to a single list using last cluster
            if isinstance(results['accuracy_history'], dict) and results['accuracy_history']:
                last_cluster = max(results['accuracy_history'].keys())
                return results['accuracy_history'][last_cluster]
            # If it's already a list, return it
            elif isinstance(results['accuracy_history'], list):
                return results['accuracy_history']
        
        return []

    def _extract_loss_history(self, results):
        """Extract loss history from results in a consistent format."""
        # Check for distillation results first
        if 'distillation_results' in results and 'val_loss_history' in results['distillation_results']:
            return results['distillation_results']['val_loss_history']
            
        # Check for cluster loss history
        if 'loss_history' in results:
            # If it's a dict with cluster IDs as keys, flatten to a single list using last cluster
            if isinstance(results['loss_history'], dict) and results['loss_history']:
                last_cluster = max(results['loss_history'].keys())
                return results['loss_history'][last_cluster]
            # If it's already a list, return it
            elif isinstance(results['loss_history'], list):
                return results['loss_history']
        
        return []

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

            # Set model based on dataset with correct input_size for HAR
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
                # Get exact feature count for HAR dataset
                import numpy as np
                try:
                    from utils.config import HAR_DIR
                    import os
                    import pandas as pd
                    har_train_file = os.path.join(HAR_DIR, "har_train.csv")
                    if os.path.exists(har_train_file):
                        # Count number of columns (excluding Activity column)
                        cols = pd.read_csv(har_train_file, nrows=1).shape[1] - 1
                        self.logger.info(f"Detected {cols} features in HAR dataset")
                        config.model_class = lambda: HARTeacher(input_size=cols)
                        config.TeacherModel = lambda: HARTeacher(input_size=cols)
                    else:
                        config.model_class = HARTeacher
                        config.TeacherModel = HARTeacher
                except Exception as e:
                    self.logger.error(f"Error determining HAR input size: {str(e)}")
                    config.model_class = HARTeacher
                    config.TeacherModel = HARTeacher

            # Now run the single experiment
            results = self.run_single_experiment(config, experiment_name)
            baseline_results[baseline][f"alpha_{alpha}"] = results
            
            # Save the results
            self.save_results(results, 'baseline', experiment_name)
            
        return baseline_results  

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

            # Set model based on dataset with correct input_size for HAR
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
                # Get exact feature count for HAR dataset
                import numpy as np
                try:
                    from utils.config import HAR_DIR
                    import os
                    import pandas as pd
                    har_train_file = os.path.join(HAR_DIR, "har_train.csv")
                    if os.path.exists(har_train_file):
                        # Count number of columns (excluding Activity column)
                        cols = pd.read_csv(har_train_file, nrows=1).shape[1] - 1
                        self.logger.info(f"Detected {cols} features in HAR dataset")
                        config.model_class = lambda: HARTeacher(input_size=cols)
                        config.TeacherModel = lambda: HARTeacher(input_size=cols)
                    else:
                        config.model_class = HARTeacher
                        config.TeacherModel = HARTeacher
                except Exception as e:
                    self.logger.error(f"Error determining HAR input size: {str(e)}")
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
            
            # Add new metrics
            summary["time_to_accuracy"] = self._calculate_time_to_accuracy(baseline_results, framework_results)
            summary["communication_efficiency"] = self._calculate_communication_efficiency(baseline_results, framework_results)
            summary["statistical_significance"] = self._calculate_statistical_significance(baseline_results, framework_results)
            
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

    def _calculate_time_to_accuracy(self, baseline_results, framework_results):
        """Calculate rounds needed to reach accuracy thresholds."""
        thresholds = [0.6, 0.7, 0.8]
        time_metrics = {}
        
        try:
            for baseline in self.baselines:
                if baseline not in baseline_results or baseline not in framework_results:
                    continue
                    
                time_metrics[baseline] = {}
                
                for alpha_key in baseline_results[baseline]:
                    if alpha_key not in framework_results[baseline]:
                        continue
                    
                    # Get accuracy histories
                    base_acc_history = baseline_results[baseline][alpha_key].get('accuracy_history', [])
                    framework_acc_history = framework_results[baseline][alpha_key].get('accuracy_history', [])
                    
                    # Skip if either history is empty
                    if not base_acc_history or not framework_acc_history:
                        continue
                    
                    time_metrics[baseline][alpha_key] = {"baseline": {}, "framework": {}}
                    
                    # Find rounds to reach threshold for baseline
                    for threshold in thresholds:
                        # Check if threshold was reached
                        above_threshold = [i for i, acc in enumerate(base_acc_history) if acc >= threshold]
                        rounds_to_threshold = above_threshold[0] + 1 if above_threshold else None
                        time_metrics[baseline][alpha_key]["baseline"][f"rounds_to_{threshold}"] = rounds_to_threshold
                    
                    # Find rounds to reach threshold for framework
                    for threshold in thresholds:
                        # Check if threshold was reached
                        above_threshold = [i for i, acc in enumerate(framework_acc_history) if acc >= threshold]
                        rounds_to_threshold = above_threshold[0] + 1 if above_threshold else None
                        time_metrics[baseline][alpha_key]["framework"][f"rounds_to_{threshold}"] = rounds_to_threshold
        except Exception as e:
            self.logger.error(f"Error in _calculate_time_to_accuracy: {str(e)}")
            
        return time_metrics

    def _calculate_communication_efficiency(self, baseline_results, framework_results):
        """Calculate accuracy per byte transmitted."""
        comm_metrics = {}
        
        try:
            for baseline in self.baselines:
                if baseline not in baseline_results or baseline not in framework_results:
                    continue
                    
                comm_metrics[baseline] = {}
                
                for alpha_key in baseline_results[baseline]:
                    if alpha_key not in framework_results[baseline]:
                        continue
                    
                    base_results = baseline_results[baseline][alpha_key]
                    framework_res = framework_results[baseline][alpha_key]
                    
                    # Get final accuracies
                    base_acc = base_results.get("final_results", {}).get("accuracy", 0)
                    framework_acc = framework_res.get("final_results", {}).get("accuracy", 0)
                    
                    # Get communication costs if available
                    base_cost = base_results.get("communication_cost", 1)  # Default to 1 to avoid division by zero
                    framework_cost = framework_res.get("communication_cost", 1) 
                    
                    # Calculate efficiency (accuracy per MB)
                    base_efficiency = base_acc / (base_cost / (1024 * 1024)) if base_cost > 0 else 0
                    framework_efficiency = framework_acc / (framework_cost / (1024 * 1024)) if framework_cost > 0 else 0
                    
                    comm_metrics[baseline][alpha_key] = {
                        "baseline_efficiency": base_efficiency,
                        "framework_efficiency": framework_efficiency,
                        "efficiency_improvement": framework_efficiency - base_efficiency
                    }
        except Exception as e:
            self.logger.error(f"Error in _calculate_communication_efficiency: {str(e)}")
            
        return comm_metrics

    def _calculate_statistical_significance(self, baseline_results, framework_results):
        """Calculate statistical significance between baseline and framework results."""
        significance_results = {}
        
        try:
            for baseline in self.baselines:
                if baseline not in baseline_results or baseline not in framework_results:
                    continue
                    
                significance_results[baseline] = {}
                
                for alpha_key in baseline_results[baseline]:
                    if alpha_key not in framework_results[baseline]:
                        continue
                    
                    # Get accuracy histories
                    base_acc_history = baseline_results[baseline][alpha_key].get('accuracy_history', [])
                    framework_acc_history = framework_results[baseline][alpha_key].get('accuracy_history', [])
                    
                    # Skip if either history is empty
                    if not base_acc_history or not framework_acc_history:
                        continue
                    
                    # If we only have one data point, use final accuracies
                    if len(base_acc_history) == 1 and len(framework_acc_history) == 1:
                        base_acc = base_acc_history[0]
                        framework_acc = framework_acc_history[0]
                        
                        # We can't compute t-test with single points, but can record improvement
                        improvement = framework_acc - base_acc
                        significance_results[baseline][alpha_key] = {
                            "t_statistic": None,
                            "p_value": None,
                            "statistically_significant": None,
                            "absolute_improvement": improvement,
                            "relative_improvement": (improvement / base_acc * 100) if base_acc > 0 else 0
                        }
                        continue
                    
                    # Match the lengths if different
                    min_len = min(len(base_acc_history), len(framework_acc_history))
                    base_acc = base_acc_history[:min_len]
                    framework_acc = framework_acc_history[:min_len]
                    
                    # Calculate t-test
                    t_stat, p_value = stats.ttest_ind(framework_acc, base_acc, equal_var=False)
                    
                    significance_results[baseline][alpha_key] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "statistically_significant": bool(p_value < 0.05)
                    }
        except Exception as e:
            self.logger.error(f"Error in _calculate_statistical_significance: {str(e)}")
            
        return significance_results

    def _get_serializable_config(self, config):
        """Convert config object to a serializable dictionary."""
        config_dict = {}
        for key, value in vars(config).items():
            # Skip non-serializable objects (like model classes, clustering objects)
            if key in ['model_class', 'TeacherModel', 'clustering']:
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

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