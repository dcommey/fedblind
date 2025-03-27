# Set environment variable to silence the joblib warning about CPU cores
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())  # Use all available logical cores

# Other imports
import json
import numpy as np
import pandas as pd
import torch
import time
import argparse
from datetime import datetime
import logging
from run_experiments import ExperimentRunner, parse_args
from utils.config import Config

def setup_logging():
    """Setup logging configuration."""
    log_dir = "lcn_results/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lcn_experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_lcn_results_dir():
    """Create and return the lcn_results directory structure."""
    base_dir = "lcn_results"
    
    # Create main directory and subdirectories
    subdirs = [
        "raw_data",           # Raw experiment results 
        "metrics",            # Processed metrics
        "tables",             # Generated tables
        "logs",               # Log files
        "models",             # Saved models
        "datasets"            # Dataset-specific results
    ]
    
    for subdir in [base_dir] + [os.path.join(base_dir, d) for d in subdirs]:
        os.makedirs(subdir, exist_ok=True)
        
    # Create dataset-specific directories
    for dataset in ["mnist", "cifar10"]:
        os.makedirs(os.path.join(base_dir, "datasets", dataset), exist_ok=True)
    
    # Create metrics-specific directories
    metrics_subdirs = ["accuracy", "convergence", "communication", "improvements", "time_to_accuracy"]
    for metric in metrics_subdirs:
        os.makedirs(os.path.join(base_dir, "metrics", metric), exist_ok=True)
    
    return base_dir

def make_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, 'tolist'):  # For torch tensors
        return obj.tolist()
    elif isinstance(obj, type):
        return str(obj)
    else:
        return str(obj)

def save_to_json(data, filepath):
    """Save data to a JSON file with error handling."""
    try:
        with open(filepath, 'w') as f:
            json.dump(make_serializable(data), f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving to {filepath}: {str(e)}")
        return False

def run_experiment(config_args, dataset, num_rounds, alpha, logger):
    """Run a single experiment with specified parameters."""
    try:
        # Create custom args
        custom_args = [
            f"--dataset={dataset}",
            f"--num_rounds={num_rounds}",
            f"--alpha={alpha}",
            "--num_clients=10",
            "--batch_size=32",
            "--learning_rate=0.01",
            "--num_clusters=2",  # For framework experiments
            "--cl_epsilon=1.0",
            "--num_quantiles=5",
            "--cluster_on=features"
        ]
        
        if config_args.use_dp_clustering:
            custom_args.append("--use_dp_clustering")

        # Add target accuracy based on dataset (directly in the args list)
        target_accuracy = 0.95 if dataset in ("mnist", "har") else 0.7
        custom_args.append(f"--target_accuracy={target_accuracy}")
        
        # Create a parser and parse custom arguments
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
        parser.add_argument("--target_accuracy", type=float, default=0.8,
                        help="Target accuracy for convergence measurement")
        
        # Parse the custom arguments
        args = parser.parse_args(custom_args)
        
        # Initialize experiment runner
        runner = ExperimentRunner(args)
        
        # Run baseline and framework experiments
        logger.info(f"Running baseline experiments for {dataset}, alpha={alpha}, rounds={num_rounds}")
        baseline_results = runner.run_baselines()
        
        logger.info(f"Running framework experiments for {dataset}, alpha={alpha}, rounds={num_rounds}")
        framework_results = runner.run_framework()
        
        # Generate summary
        logger.info(f"Generating summary for {dataset}, alpha={alpha}")
        summary = runner.generate_summary(baseline_results, framework_results)
        
        return {
            "baseline_results": baseline_results,
            "framework_results": framework_results,
            "summary": summary,
            "config": {
                "dataset": dataset,
                "num_rounds": num_rounds,
                "alpha": alpha,
                "num_clients": 10
            }
        }
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "config": {
                "dataset": dataset,
                "num_rounds": num_rounds,
                "alpha": alpha,
                "num_clients": 10
            }
        }

def save_experiment_results(results, base_dir, dataset, alpha):
    """Save raw experiment results and extract metrics."""
    # Save raw data
    raw_data_file = os.path.join(base_dir, "raw_data", f"{dataset}_alpha_{alpha}.json")
    save_to_json(results, raw_data_file)
    
    # Extract and save accuracy metrics
    accuracies = extract_accuracy_metrics(results)
    acc_file = os.path.join(base_dir, "metrics", "accuracy", f"{dataset}_alpha_{alpha}.json")
    save_to_json(accuracies, acc_file)
    
    # Extract and save convergence metrics
    convergence = extract_convergence_metrics(results)
    conv_file = os.path.join(base_dir, "metrics", "convergence", f"{dataset}_alpha_{alpha}.json")
    save_to_json(convergence, conv_file)
    
    # Extract and save communication efficiency
    comm_eff = extract_communication_metrics(results)
    comm_file = os.path.join(base_dir, "metrics", "communication", f"{dataset}_alpha_{alpha}.json")
    save_to_json(comm_eff, comm_file)
    
    # Extract and save improvements
    improvements = extract_improvement_metrics(results)
    imp_file = os.path.join(base_dir, "metrics", "improvements", f"{dataset}_alpha_{alpha}.json")
    save_to_json(improvements, imp_file)
    
    # Extract and save time to accuracy
    time_to_acc = extract_time_to_accuracy(results)
    time_file = os.path.join(base_dir, "metrics", "time_to_accuracy", f"{dataset}_alpha_{alpha}.json")
    save_to_json(time_to_acc, time_file)
    
    # Save CSV files for each metric for easy table generation
    save_metric_to_csv(accuracies, base_dir, "tables", f"{dataset}_alpha_{alpha}_accuracy.csv")
    save_metric_to_csv(convergence, base_dir, "tables", f"{dataset}_alpha_{alpha}_convergence.csv")
    save_metric_to_csv(comm_eff, base_dir, "tables", f"{dataset}_alpha_{alpha}_communication.csv")
    save_metric_to_csv(improvements, base_dir, "tables", f"{dataset}_alpha_{alpha}_improvements.csv")
    save_metric_to_csv(time_to_acc, base_dir, "tables", f"{dataset}_alpha_{alpha}_time_to_accuracy.csv")
    
    return {
        "accuracy": accuracies,
        "convergence": convergence,
        "communication": comm_eff,
        "improvements": improvements,
        "time_to_accuracy": time_to_acc
    }

def extract_accuracy_metrics(results):
    """Extract accuracy metrics from results."""
    accuracy_metrics = {"baseline": {}, "framework": {}}
    
    # Extract baseline accuracies
    for baseline, baseline_data in results.get("baseline_results", {}).items():
        for alpha_key, data in baseline_data.items():
            if "final_results" in data and "accuracy" in data["final_results"]:
                accuracy_metrics["baseline"][f"{baseline}_{alpha_key}"] = data["final_results"]["accuracy"]
    
    # Extract framework accuracies
    for baseline, framework_data in results.get("framework_results", {}).items():
        for alpha_key, data in framework_data.items():
            if "final_results" in data and "accuracy" in data["final_results"]:
                accuracy_metrics["framework"][f"{baseline}_{alpha_key}"] = data["final_results"]["accuracy"]
    
    return accuracy_metrics

def extract_convergence_metrics(results):
    """Extract model convergence metrics from results."""
    convergence_metrics = {"baseline": {}, "framework": {}}
    
    # Get summary data if available
    summary = results.get("summary", {})
    
    # Extract baseline convergence
    for baseline in summary.get("baseline_summary", {}):
        for alpha_key, data in summary["baseline_summary"][baseline].items():
            rounds_to_target = data.get("rounds_to_target")
            convergence_metrics["baseline"][f"{baseline}_{alpha_key}"] = {
                "rounds_to_target": rounds_to_target,
                "converged": rounds_to_target is not None
            }
    
    # Extract framework convergence
    for baseline in summary.get("framework_summary", {}):
        for alpha_key, data in summary["framework_summary"][baseline].items():
            rounds_to_target = data.get("rounds_to_target")
            convergence_metrics["framework"][f"{baseline}_{alpha_key}"] = {
                "rounds_to_target": rounds_to_target,
                "converged": rounds_to_target is not None
            }
    
    return convergence_metrics

def extract_communication_metrics(results):
    """Extract communication efficiency metrics from results."""
    comm_metrics = {}
    
    # Get communication efficiency metrics from summary if available
    summary = results.get("summary", {})
    comm_eff = summary.get("communication_efficiency", {})
    
    # Structure metrics in a comparable format
    for baseline in comm_eff:
        for alpha_key in comm_eff[baseline]:
            if baseline not in comm_metrics:
                comm_metrics[baseline] = {}
            
            baseline_eff = comm_eff[baseline][alpha_key].get("baseline_efficiency", 0)
            framework_eff = comm_eff[baseline][alpha_key].get("framework_efficiency", 0)
            improvement = comm_eff[baseline][alpha_key].get("efficiency_improvement", 0)
            
            comm_metrics[baseline][alpha_key] = {
                "baseline_efficiency": baseline_eff,
                "framework_efficiency": framework_eff,
                "improvement": improvement,
                "improvement_percentage": (improvement / baseline_eff * 100) if baseline_eff > 0 else 0
            }
    
    return comm_metrics

def extract_improvement_metrics(results):
    """Extract improvement metrics from results."""
    improvement_metrics = {}
    
    # Get improvements from summary
    summary = results.get("summary", {})
    improvements = summary.get("improvements", {})
    
    # Structure metrics for easy comparison
    for baseline in improvements:
        improvement_metrics[baseline] = {}
        for alpha_key in improvements[baseline]:
            improvement_percentage = improvements[baseline][alpha_key].get("accuracy_improvement_percentage", 0)
            base_acc = improvements[baseline][alpha_key].get("base_accuracy", 0)
            framework_acc = improvements[baseline][alpha_key].get("framework_accuracy", 0)
            
            improvement_metrics[baseline][alpha_key] = {
                "accuracy_improvement": framework_acc - base_acc,
                "accuracy_improvement_percentage": improvement_percentage,
                "base_accuracy": base_acc,
                "framework_accuracy": framework_acc
            }
    
    return improvement_metrics

def extract_time_to_accuracy(results):
    """Extract time to accuracy metrics from results."""
    time_metrics = {}
    
    # Get time to accuracy metrics from summary
    summary = results.get("summary", {})
    time_to_acc = summary.get("time_to_accuracy", {})
    
    # Structure for easy comparison
    for baseline in time_to_acc:
        time_metrics[baseline] = {}
        for alpha_key in time_to_acc[baseline]:
            baseline_times = time_to_acc[baseline][alpha_key].get("baseline", {})
            framework_times = time_to_acc[baseline][alpha_key].get("framework", {})
            
            thresholds = []
            for key in baseline_times:
                if key.startswith("rounds_to_"):
                    threshold = key.replace("rounds_to_", "")
                    thresholds.append(threshold)
            
            threshold_metrics = {}
            for threshold in thresholds:
                baseline_rounds = baseline_times.get(f"rounds_to_{threshold}")
                framework_rounds = framework_times.get(f"rounds_to_{threshold}")
                
                if baseline_rounds is not None and framework_rounds is not None:
                    rounds_saved = baseline_rounds - framework_rounds
                    percentage_saved = (rounds_saved / baseline_rounds * 100) if baseline_rounds > 0 else 0
                    
                    threshold_metrics[threshold] = {
                        "baseline_rounds": baseline_rounds,
                        "framework_rounds": framework_rounds,
                        "rounds_saved": rounds_saved,
                        "percentage_saved": percentage_saved
                    }
            
            time_metrics[baseline][alpha_key] = threshold_metrics
    
    return time_metrics

def save_metric_to_csv(metric_data, base_dir, subdir, filename):
    """Save metric data to CSV for easy table generation."""
    try:
        # Convert nested dictionary to DataFrame
        rows = []
        
        # Handle different metric structures
        if "baseline" in metric_data and "framework" in metric_data:
            # Accuracy-like structure
            for model_type in ["baseline", "framework"]:
                for model_key, value in metric_data[model_type].items():
                    parts = model_key.split('_')
                    algorithm = parts[0]
                    alpha = '_'.join(parts[1:])
                    
                    rows.append({
                        "Type": model_type,
                        "Algorithm": algorithm,
                        "Alpha": alpha,
                        "Value": value
                    })
        else:
            # Improvement-like nested structure
            for algorithm in metric_data:
                for alpha_key, values in metric_data[algorithm].items():
                    if isinstance(values, dict):
                        for metric_name, metric_value in values.items():
                            rows.append({
                                "Algorithm": algorithm,
                                "Alpha": alpha_key,
                                "Metric": metric_name,
                                "Value": metric_value
                            })
                    else:
                        rows.append({
                            "Algorithm": algorithm,
                            "Alpha": alpha_key,
                            "Value": values
                        })
        
        # Create DataFrame and save
        if rows:
            df = pd.DataFrame(rows)
            filepath = os.path.join(base_dir, subdir, filename)
            df.to_csv(filepath, index=False)
            return True
        return False
    except Exception as e:
        logging.error(f"Error saving metric to CSV: {str(e)}")
        return False

def create_combined_data_tables(base_dir):
    """Create combined data tables across all experiments."""
    try:
        # Directories to search for CSV files
        table_dir = os.path.join(base_dir, "tables")
        
        # Group CSV files by metric type
        metric_files = {
            "accuracy": [],
            "convergence": [],
            "communication": [],
            "improvements": [],
            "time_to_accuracy": []
        }
        
        for filename in os.listdir(table_dir):
            if filename.endswith(".csv"):
                for metric in metric_files:
                    if metric in filename:
                        metric_files[metric].append(os.path.join(table_dir, filename))
        
        # Combine CSVs for each metric
        for metric, files in metric_files.items():
            if not files:
                continue
                
            combined_data = []
            for file in files:
                df = pd.read_csv(file)
                # Extract dataset and alpha from filename
                filename = os.path.basename(file)
                parts = filename.split('_')
                dataset = parts[0]
                alpha = parts[2].replace('.csv', '')
                
                # Add dataset and alpha columns
                df['Dataset'] = dataset
                df['Alpha'] = alpha
                
                combined_data.append(df)
            
            if combined_data:
                # Combine all dataframes
                combined_df = pd.concat(combined_data, ignore_index=True)
                
                # Save combined data
                combined_file = os.path.join(base_dir, "tables", f"combined_{metric}.csv")
                combined_df.to_csv(combined_file, index=False)
                
                logging.info(f"Created combined table for {metric}: {combined_file}")
        
        return True
    except Exception as e:
        logging.error(f"Error creating combined data tables: {str(e)}")
        return False

def run_all_experiments():
    """Run all experiments and save results."""
    # Setup
    logger = setup_logging()
    logger.info("Starting LCN experiments")
    
    # Create results directory
    base_dir = create_lcn_results_dir()
    logger.info(f"Created results directory: {base_dir}")
    
    # Define experiment configurations
    experiments = [
        # {"dataset": "cifar10", "num_rounds": 150, "alpha": 0.5},
        # {"dataset": "cifar10", "num_rounds": 150, "alpha": 1.0},
        # {"dataset": "cifar10", "num_rounds": 150, "alpha": 3.0},
        {"dataset": "mnist", "num_rounds": 1, "alpha": 0.5},
        {"dataset": "mnist", "num_rounds": 1, "alpha": 1.0},
        {"dataset": "mnist", "num_rounds": 1, "alpha": 3.0}
    ]
    
    # Parse command line arguments for custom configuration
    parser = argparse.ArgumentParser(description="Run LCN Experiments")
    parser.add_argument("--use_dp_clustering", action='store_true', help="Use differentially private clustering")
    config_args = parser.parse_args()
    
    # Track all metrics
    all_metrics = {}
    
    # Run experiments
    for i, exp in enumerate(experiments):
        logger.info(f"Running experiment {i+1}/{len(experiments)}: "
                   f"dataset={exp['dataset']}, alpha={exp['alpha']}, rounds={exp['num_rounds']}")
        
        # Run experiment
        start_time = time.time()
        results = run_experiment(config_args, exp["dataset"], exp["num_rounds"], exp["alpha"], logger)
        duration = time.time() - start_time
        
        # Save experiment results
        if "error" not in results:
            logger.info(f"Experiment completed in {duration:.2f} seconds. Saving results...")
            metrics = save_experiment_results(results, base_dir, exp["dataset"], exp["alpha"])
            
            # Store metrics for summary
            key = f"{exp['dataset']}_alpha_{exp['alpha']}"
            all_metrics[key] = metrics
        else:
            logger.error(f"Experiment failed: {results['error']}")
    
    # Create combined data tables
    logger.info("Creating combined data tables...")
    create_combined_data_tables(base_dir)
    
    # Save summary of all experiments
    summary_file = os.path.join(base_dir, "experiment_summary.json")
    save_to_json({
        "experiments": experiments,
        "metrics_overview": all_metrics
    }, summary_file)
    
    logger.info(f"All experiments completed. Results saved in {base_dir}")
    logger.info("Run analyze_lcn_results.py to generate plots and LaTeX tables")
    
    return all_metrics

if __name__ == "__main__":
    try:
        metrics = run_all_experiments()
        print(f"Experiments completed successfully. Results saved in lcn_results directory.")
        print(f"Run 'python analyze_lcn_results.py' to generate plots and LaTeX tables.")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"Error running experiments: {str(e)}")
