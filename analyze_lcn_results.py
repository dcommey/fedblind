import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse
from pathlib import Path
import re
import logging
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (8, 6)
})

# Use seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
sns.set_context("paper")

def setup_logging():
    """Setup logging configuration."""
    log_dir = "lcn_results/analysis_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "analysis.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_json_data(filepath):
    """Load data from a JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {str(e)}")
        return None

def create_output_dirs(base_dir="lcn_results"):
    """Create output directories for plots and LaTeX tables."""
    plots_dir = os.path.join(base_dir, "plots")
    latex_dir = os.path.join(base_dir, "latex_tables")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    
    # Create subdirectories for different types of plots
    plot_subdirs = ["accuracy", "learning_curves", "convergence", "communication", 
                    "improvements", "time_to_accuracy", "comparison"]
    
    for subdir in plot_subdirs:
        os.makedirs(os.path.join(plots_dir, subdir), exist_ok=True)
    
    return plots_dir, latex_dir

def load_experiment_data(base_dir="lcn_results"):
    """Load experiment data from the organized directory structure."""
    experiment_data = {
        "raw_data": {},
        "metrics": {
            "accuracy": {},
            "convergence": {},
            "communication": {},
            "improvements": {},
            "time_to_accuracy": {}
        },
        "tables": {},
        "summary": None
    }
    
    # Load raw data
    raw_data_dir = os.path.join(base_dir, "raw_data")
    if os.path.exists(raw_data_dir):
        for filename in os.listdir(raw_data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(raw_data_dir, filename)
                key = filename.replace(".json", "")
                data = load_json_data(filepath)
                if data:
                    experiment_data["raw_data"][key] = data
    
    # Load metrics data
    metrics_dir = os.path.join(base_dir, "metrics")
    if os.path.exists(metrics_dir):
        for metric_type in experiment_data["metrics"]:
            metric_dir = os.path.join(metrics_dir, metric_type)
            if os.path.exists(metric_dir):
                for filename in os.listdir(metric_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(metric_dir, filename)
                        key = filename.replace(".json", "")
                        data = load_json_data(filepath)
                        if data:
                            experiment_data["metrics"][metric_type][key] = data
    
    # Load table data
    tables_dir = os.path.join(base_dir, "tables")
    if os.path.exists(tables_dir):
        for filename in os.listdir(tables_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(tables_dir, filename)
                key = filename.replace(".csv", "")
                try:
                    data = pd.read_csv(filepath)
                    experiment_data["tables"][key] = data
                except Exception as e:
                    logging.error(f"Error loading table {filepath}: {str(e)}")
    
    # Load summary data
    summary_file = os.path.join(base_dir, "experiment_summary.json")
    if os.path.exists(summary_file):
        experiment_data["summary"] = load_json_data(summary_file)
    
    return experiment_data

def generate_accuracy_plots(data, plots_dir):
    """Generate accuracy comparison plots for all datasets and alphas."""
    logging.info("Generating accuracy comparison plots...")
    
    # Create a multi-page PDF for all accuracy plots
    pdf_path = os.path.join(plots_dir, "accuracy", "all_accuracy_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        for key, accuracy_data in data["metrics"]["accuracy"].items():
            try:
                match = re.match(r'(.+)_alpha_(.+)', key)
                if not match:
                    continue
                    
                dataset, alpha = match.groups()
                
                # Extract algorithms and their accuracies
                algorithms = set()
                for model_type in ["baseline", "framework"]:
                    for model_key in accuracy_data.get(model_type, {}):
                        algorithm = model_key.split('_')[0]
                        algorithms.add(algorithm)
                
                algorithms = sorted(list(algorithms))
                
                baseline_accs = []
                framework_accs = []
                
                for algorithm in algorithms:
                    baseline_key = f"{algorithm}_alpha_{alpha}"
                    framework_key = f"{algorithm}_alpha_{alpha}"
                    
                    baseline_acc = accuracy_data.get("baseline", {}).get(baseline_key, 0)
                    framework_acc = accuracy_data.get("framework", {}).get(framework_key, 0)
                    
                    baseline_accs.append(baseline_acc)
                    framework_accs.append(framework_acc)
                
                # Create a new figure for each dataset and alpha
                fig, ax = plt.subplots(figsize=(8, 6))
                x = np.arange(len(algorithms))
                width = 0.35
                
                # Plot bars
                rects1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline', color='#3274A1')
                rects2 = ax.bar(x + width/2, framework_accs, width, label='Framework', color='#E1812C')
                
                # Add value labels
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.3f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=9)
                
                autolabel(rects1)
                autolabel(rects2)
                
                # Add labels and title
                ax.set_xlabel('Algorithm')
                ax.set_ylabel('Accuracy')
                dataset_display = dataset.upper() if dataset.lower() in ['mnist', 'cifar10'] else dataset.capitalize()
                ax.set_title(f'Accuracy Comparison - {dataset_display} (α={alpha})')
                ax.set_xticks(x)
                ax.set_xticklabels([alg.upper() for alg in algorithms])
                ax.legend()
                
                # Set y-axis to start slightly below 0 for better visualization
                ax.set_ylim(0, max(max(baseline_accs), max(framework_accs)) * 1.1)
                
                # Add grid for readability
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Adjust layout and save the figure
                plt.tight_layout()
                
                # Save to the PDF
                pdf.savefig(fig)
                
                # Also save individual files
                individual_file = os.path.join(plots_dir, "accuracy", f"{dataset}_alpha_{alpha}_accuracy.pdf")
                fig.savefig(individual_file)
                
                plt.close(fig)
                
                logging.info(f"Generated accuracy plot for {dataset} (alpha={alpha})")
            except Exception as e:
                logging.error(f"Error generating accuracy plot for {key}: {str(e)}")
    
    logging.info(f"All accuracy plots saved to {pdf_path}")

def generate_learning_curves(data, plots_dir):
    """Generate learning curves for all experiments."""
    logging.info("Generating learning curves...")
    
    # Create a multi-page PDF for all learning curves
    pdf_path = os.path.join(plots_dir, "learning_curves", "all_learning_curves.pdf")
    with PdfPages(pdf_path) as pdf:
        for key, raw_data in data["raw_data"].items():
            try:
                match = re.match(r'(.+)_alpha_(.+)', key)
                if not match:
                    continue
                    
                dataset, alpha = match.groups()
                
                # Get baseline and framework results
                baseline_results = raw_data.get("baseline_results", {})
                framework_results = raw_data.get("framework_results", {})
                
                for algorithm in baseline_results:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Extract accuracy histories
                    alpha_key = list(baseline_results[algorithm].keys())[0]
                    
                    base_history = baseline_results[algorithm][alpha_key].get("accuracy_history", [])
                    framework_history = framework_results.get(algorithm, {}).get(alpha_key, {}).get("accuracy_history", [])
                    
                    # Plot histories
                    if base_history:
                        ax.plot(base_history, label=f'{algorithm.upper()} Baseline', 
                               marker='o', markersize=4, markevery=5, linewidth=2)
                    if framework_history:
                        ax.plot(framework_history, label=f'{algorithm.upper()} Framework', 
                               marker='s', markersize=4, markevery=5, linewidth=2)
                    
                    # Add labels and title
                    ax.set_xlabel('Rounds')
                    ax.set_ylabel('Accuracy')
                    dataset_display = dataset.upper() if dataset.lower() in ['mnist', 'cifar10'] else dataset.capitalize()
                    ax.set_title(f'Learning Curves - {dataset_display} (α={alpha}) - {algorithm.upper()}')
                    ax.legend(loc='lower right')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Force integer x-ticks for rounds
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    
                    # Add a horizontal line at 0.8 accuracy for reference
                    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='0.8 Accuracy')
                    
                    # Save to the PDF
                    plt.tight_layout()
                    pdf.savefig(fig)
                    
                    # Also save individual file
                    individual_file = os.path.join(plots_dir, "learning_curves", 
                                                  f"{dataset}_alpha_{alpha}_{algorithm}_learning_curve.pdf")
                    fig.savefig(individual_file)
                    
                    plt.close(fig)
                    
                    logging.info(f"Generated learning curve for {dataset} (alpha={alpha}) - {algorithm}")
            except Exception as e:
                logging.error(f"Error generating learning curve for {key}: {str(e)}")
    
    logging.info(f"All learning curves saved to {pdf_path}")

def generate_time_to_accuracy_plots(data, plots_dir):
    """Generate time-to-accuracy comparison plots."""
    logging.info("Generating time-to-accuracy plots...")
    
    # Create a multi-page PDF for all time-to-accuracy plots
    pdf_path = os.path.join(plots_dir, "time_to_accuracy", "all_time_to_accuracy_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        for key, time_data in data["metrics"]["time_to_accuracy"].items():
            try:
                match = re.match(r'(.+)_alpha_(.+)', key)
                if not match:
                    continue
                    
                dataset, alpha = match.groups()
                
                # For each algorithm, generate a bar chart of time to reach different accuracy thresholds
                for algorithm in time_data:
                    # Skip if no data for this algorithm
                    if not time_data[algorithm]:
                        continue
                    
                    # Get all thresholds from the first alpha key
                    alpha_key = list(time_data[algorithm].keys())[0]
                    threshold_data = time_data[algorithm][alpha_key]
                    
                    # Extract thresholds and rounds
                    thresholds = sorted([float(t) for t in threshold_data.keys()])
                    
                    baseline_rounds = []
                    framework_rounds = []
                    labels = []
                    
                    for threshold in thresholds:
                        threshold_str = str(threshold)
                        if threshold_str in threshold_data:
                            baseline_round = threshold_data[threshold_str].get("baseline_rounds")
                            framework_round = threshold_data[threshold_str].get("framework_rounds")
                            
                            if baseline_round is not None and framework_round is not None:
                                baseline_rounds.append(baseline_round)
                                framework_rounds.append(framework_round)
                                labels.append(f"{threshold:.1f}")
                    
                    if not baseline_rounds or not framework_rounds:
                        continue
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    x = np.arange(len(labels))
                    width = 0.35
                    
                    rects1 = ax.bar(x - width/2, baseline_rounds, width, label='Baseline', color='#3274A1')
                    rects2 = ax.bar(x + width/2, framework_rounds, width, label='Framework', color='#E1812C')
                    
                    # Add value labels
                    def autolabel(rects):
                        for rect in rects:
                            height = rect.get_height()
                            ax.annotate(f'{int(height)}',
                                      xy=(rect.get_x() + rect.get_width() / 2, height),
                                      xytext=(0, 3),
                                      textcoords="offset points",
                                      ha='center', va='bottom',
                                      fontsize=9)
                    
                    autolabel(rects1)
                    autolabel(rects2)
                    
                    # Add labels and title
                    ax.set_xlabel('Accuracy Threshold')
                    ax.set_ylabel('Rounds to Reach Threshold')
                    dataset_display = dataset.upper() if dataset.lower() in ['mnist', 'cifar10'] else dataset.capitalize()
                    ax.set_title(f'Time to Accuracy - {dataset_display} (α={alpha}) - {algorithm.upper()}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    ax.legend()
                    
                    # Add grid for readability
                    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                    
                    # Set y-axis to start at 0
                    ax.set_ylim(0, max(max(baseline_rounds), max(framework_rounds)) * 1.1)
                    
                    # Save to the PDF
                    plt.tight_layout()
                    pdf.savefig(fig)
                    
                    # Also save individual file
                    individual_file = os.path.join(plots_dir, "time_to_accuracy", 
                                                  f"{dataset}_alpha_{alpha}_{algorithm}_time_to_accuracy.pdf")
                    fig.savefig(individual_file)
                    
                    plt.close(fig)
                    
                    logging.info(f"Generated time-to-accuracy plot for {dataset} (alpha={alpha}) - {algorithm}")
            except Exception as e:
                logging.error(f"Error generating time-to-accuracy plot for {key}: {str(e)}")
    
    logging.info(f"All time-to-accuracy plots saved to {pdf_path}")

def generate_improvement_heatmap(data, plots_dir):
    """Generate improvement heatmaps for accuracy improvements across datasets and algorithms."""
    logging.info("Generating improvement heatmaps...")
    
    # Prepare data for heatmap
    try:
        # Create a dictionary mapping (dataset, alpha) -> algorithm -> improvement
        improvements_data = {}
        
        for key, improvement_data in data["metrics"]["improvements"].items():
            match = re.match(r'(.+)_alpha_(.+)', key)
            if not match:
                continue
                
            dataset, alpha = match.groups()
            dataset_key = f"{dataset.upper()} (α={alpha})"
            
            improvements_data[dataset_key] = {}
            
            # Extract improvement percentages for each algorithm
            for algorithm, alpha_data in improvement_data.items():
                for alpha_key, values in alpha_data.items():
                    if "accuracy_improvement_percentage" in values:
                        improvements_data[dataset_key][algorithm.upper()] = values["accuracy_improvement_percentage"]
        
        if not improvements_data:
            logging.warning("No improvement data found for heatmap")
            return
        
        # Convert to DataFrame for heatmap
        datasets = sorted(improvements_data.keys())
        algorithms = sorted(set().union(*[set(improvements_data[d].keys()) for d in datasets]))
        
        improvement_matrix = np.zeros((len(datasets), len(algorithms)))
        
        for i, dataset in enumerate(datasets):
            for j, algorithm in enumerate(algorithms):
                if algorithm in improvements_data[dataset]:
                    improvement_matrix[i, j] = improvements_data[dataset][algorithm]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = sns.heatmap(improvement_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                   xticklabels=algorithms, yticklabels=datasets, ax=ax)
        
        # Add a colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label("Accuracy Improvement (%)")
        
        ax.set_title("Accuracy Improvement Percentage by Algorithm and Dataset")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Dataset")
        
        # Save the plot
        plt.tight_layout()
        heatmap_file = os.path.join(plots_dir, "improvements", "accuracy_improvement_heatmap.pdf")
        fig.savefig(heatmap_file)
        
        plt.close(fig)
        
        logging.info(f"Generated improvement heatmap: {heatmap_file}")
    except Exception as e:
        logging.error(f"Error generating improvement heatmap: {str(e)}")

def generate_algorithm_comparison_plot(data, plots_dir):
    """Generate algorithm comparison plot across datasets and alphas."""
    logging.info("Generating algorithm comparison plot...")
    
    try:
        # Collect algorithm performance across datasets and alphas
        algorithm_data = {}
        
        for key, accuracy_data in data["metrics"]["accuracy"].items():
            match = re.match(r'(.+)_alpha_(.+)', key)
            if not match:
                continue
                
            dataset, alpha = match.groups()
            
            # Extract algorithms and their accuracies
            for model_type in ["baseline", "framework"]:
                if model_type not in accuracy_data:
                    continue
                    
                for model_key, accuracy in accuracy_data[model_type].items():
                    parts = model_key.split('_')
                    algorithm = parts[0].upper()
                    
                    if algorithm not in algorithm_data:
                        algorithm_data[algorithm] = {"baseline": [], "framework": []}
                    
                    algorithm_data[algorithm][model_type].append(accuracy)
        
        if not algorithm_data:
            logging.warning("No algorithm data found for comparison plot")
            return
        
        # Create boxplots comparing baseline and framework performance for each algorithm
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        algorithms = sorted(algorithm_data.keys())
        baseline_data = [algorithm_data[alg]["baseline"] for alg in algorithms]
        framework_data = [algorithm_data[alg]["framework"] for alg in algorithms]
        
        # Set positions and width for box plots
        positions = np.arange(len(algorithms)) * 2.5
        width = 0.8
        
        # Create box plots
        bp1 = ax.boxplot(baseline_data, positions=positions-width/2, widths=width, 
                        patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(framework_data, positions=positions+width/2, widths=width, 
                        patch_artist=True, showfliers=False)
        
        # Set colors for box plots
        for box in bp1['boxes']:
            box.set(facecolor='#3274A1', alpha=0.7)
        for box in bp2['boxes']:
            box.set(facecolor='#E1812C', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Accuracy')
        ax.set_title('Algorithm Performance Comparison: Baseline vs Framework')
        ax.set_xticks(positions)
        ax.set_xticklabels(algorithms)
        
        # Add a legend
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Baseline', 'Framework'], loc='upper left')
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Save the plot
        plt.tight_layout()
        comparison_file = os.path.join(plots_dir, "comparison", "algorithm_performance_comparison.pdf")
        fig.savefig(comparison_file)
        
        plt.close(fig)
        
        logging.info(f"Generated algorithm comparison plot: {comparison_file}")
    except Exception as e:
        logging.error(f"Error generating algorithm comparison plot: {str(e)}")

def generate_latex_tables(data, latex_dir):
    """Generate LaTeX tables from the data."""
    logging.info("Generating LaTeX tables...")
    
    # Generate accuracy table
    generate_accuracy_latex_table(data, latex_dir)
    
    # Generate time-to-accuracy table
    generate_time_to_accuracy_latex_table(data, latex_dir)
    
    # Generate improvement table
    generate_improvement_latex_table(data, latex_dir)
    
    # Generate communication efficiency table
    generate_communication_latex_table(data, latex_dir)

def dataframe_to_latex(df, caption, label, float_format="{:.3f}"):
    """Convert a pandas DataFrame to a LaTeX table string."""
    latex_table = df.to_latex(index=False, float_format=float_format.format, escape=False)
    
    # Add caption and label
    latex_table = latex_table.replace("\\begin{table}", "\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:" + label + "}")
    
    # Center the table
    latex_table = latex_table.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")
    
    return latex_table

def generate_accuracy_latex_table(data, latex_dir):
    """Generate LaTeX table for accuracy comparison."""
    try:
        # Prepare data structure for the table
        accuracy_rows = []
        
        for key, accuracy_data in data["metrics"]["accuracy"].items():
            match = re.match(r'(.+)_alpha_(.+)', key)
            if not match:
                continue
                
            dataset, alpha = match.groups()
            
            for model_type in ["baseline", "framework"]:
                if model_type not in accuracy_data:
                    continue
                    
                for model_key, accuracy in accuracy_data[model_type].items():
                    parts = model_key.split('_')
                    algorithm = parts[0].upper()
                    
                    row = {
                        "Dataset": dataset.upper(),
                        "Alpha": alpha,
                        "Algorithm": algorithm,
                        "Type": model_type.capitalize(),
                        "Accuracy": accuracy
                    }
                    
                    accuracy_rows.append(row)
        
        if not accuracy_rows:
            logging.warning("No accuracy data found for LaTeX table")
            return
        
        # Create DataFrame
        df = pd.DataFrame(accuracy_rows)
        
        # Pivot table for better presentation
        pivot_df = df.pivot_table(
            index=["Dataset", "Alpha", "Algorithm"],
            columns=["Type"],
            values=["Accuracy"]
        )
        
        # Flatten the column hierarchy
        pivot_df.columns = [col[1] for col in pivot_df.columns]
        
        # Calculate improvement
        pivot_df["Improvement (%)"] = ((pivot_df["Framework"] - pivot_df["Baseline"]) / pivot_df["Baseline"]) * 100
        
        # Reset index for better formatting
        pivot_df = pivot_df.reset_index()
        
        # Rename columns
        pivot_df = pivot_df[["Dataset", "Alpha", "Algorithm", "Baseline", "Framework", "Improvement (%)"]]
        
        # Generate LaTeX table
        caption = "Accuracy Comparison between Baseline and Framework"
        label = "accuracy_comparison"
        latex_table = dataframe_to_latex(pivot_df, caption, label)
        
        # Save to file
        latex_file = os.path.join(latex_dir, "accuracy_comparison.tex")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        logging.info(f"Generated accuracy LaTeX table: {latex_file}")
    except Exception as e:
        logging.error(f"Error generating accuracy LaTeX table: {str(e)}")

def generate_time_to_accuracy_latex_table(data, latex_dir):
    """Generate LaTeX table for time-to-accuracy comparison."""
    try:
        # Prepare data structure for the table
        time_rows = []
        
        for key, time_data in data["metrics"]["time_to_accuracy"].items():
            match = re.match(r'(.+)_alpha_(.+)', key)
            if not match:
                continue
                
            dataset, alpha = match.groups()
            
            for algorithm in time_data:
                if not time_data[algorithm]:
                    continue
                
                # Get all thresholds from the first alpha key
                alpha_key = list(time_data[algorithm].keys())[0]
                threshold_data = time_data[algorithm][alpha_key]
                
                for threshold, values in threshold_data.items():
                    baseline_rounds = values.get("baseline_rounds")
                    framework_rounds = values.get("framework_rounds")
                    rounds_saved = values.get("rounds_saved")
                    percentage_saved = values.get("percentage_saved")
                    
                    if baseline_rounds is not None and framework_rounds is not None:
                        row = {
                            "Dataset": dataset.upper(),
                            "Alpha": alpha,
                            "Algorithm": algorithm.upper(),
                            "Threshold": float(threshold),
                            "Baseline Rounds": baseline_rounds,
                            "Framework Rounds": framework_rounds,
                            "Rounds Saved": rounds_saved,
                            "Improvement (%)": percentage_saved
                        }
                        
                        time_rows.append(row)
        
        if not time_rows:
            logging.warning("No time-to-accuracy data found for LaTeX table")
            return
        
        # Create DataFrame
        df = pd.DataFrame(time_rows)
        
        # Sort by dataset, alpha, algorithm, threshold
        df = df.sort_values(by=["Dataset", "Alpha", "Algorithm", "Threshold"])
        
        # Generate LaTeX table
        caption = "Time to Accuracy: Rounds Required to Reach Accuracy Thresholds"
        label = "time_to_accuracy"
        latex_table = dataframe_to_latex(df, caption, label, float_format="{:.1f}")
        
        # Save to file
        latex_file = os.path.join(latex_dir, "time_to_accuracy.tex")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        logging.info(f"Generated time-to-accuracy LaTeX table: {latex_file}")
    except Exception as e:
        logging.error(f"Error generating time-to-accuracy LaTeX table: {str(e)}")

def generate_improvement_latex_table(data, latex_dir):
    """Generate LaTeX table for improvement metrics."""
    try:
        # Prepare data structure for the table
        improvement_rows = []
        
        for key, improvement_data in data["metrics"]["improvements"].items():
            match = re.match(r'(.+)_alpha_(.+)', key)
            if not match:
                continue
                
            dataset, alpha = match.groups()
            
            for algorithm, alpha_data in improvement_data.items():
                for alpha_key, values in alpha_data.items():
                    if "accuracy_improvement_percentage" in values:
                        row = {
                            "Dataset": dataset.upper(),
                            "Alpha": alpha,
                            "Algorithm": algorithm.upper(),
                            "Baseline Accuracy": values.get("base_accuracy", 0),
                            "Framework Accuracy": values.get("framework_accuracy", 0),
                            "Absolute Improvement": values.get("accuracy_improvement", 0),
                            "Relative Improvement (%)": values.get("accuracy_improvement_percentage", 0)
                        }
                        
                        improvement_rows.append(row)
        
        if not improvement_rows:
            logging.warning("No improvement data found for LaTeX table")
            return
        
        # Create DataFrame
        df = pd.DataFrame(improvement_rows)
        
        # Sort by dataset, alpha, algorithm
        df = df.sort_values(by=["Dataset", "Alpha", "Algorithm"])
        
        # Generate LaTeX table
        caption = "Performance Improvements of Framework over Baseline"
        label = "performance_improvements"
        latex_table = dataframe_to_latex(df, caption, label, float_format="{:.2f}")
        
        # Save to file
        latex_file = os.path.join(latex_dir, "performance_improvements.tex")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        logging.info(f"Generated improvement LaTeX table: {latex_file}")
    except Exception as e:
        logging.error(f"Error generating improvement LaTeX table: {str(e)}")

def generate_communication_latex_table(data, latex_dir):
    """Generate LaTeX table for communication efficiency."""
    try:
        # Prepare data structure for the table
        comm_rows = []
        
        for key, comm_data in data["metrics"]["communication"].items():
            match = re.match(r'(.+)_alpha_(.+)', key)
            if not match:
                continue
                
            dataset, alpha = match.groups()
            
            for algorithm, alpha_data in comm_data.items():
                for alpha_key, values in alpha_data.items():
                    row = {
                        "Dataset": dataset.upper(),
                        "Alpha": alpha,
                        "Algorithm": algorithm.upper(),
                        "Baseline Efficiency": values.get("baseline_efficiency", 0),
                        "Framework Efficiency": values.get("framework_efficiency", 0),
                        "Efficiency Improvement": values.get("improvement", 0),
                        "Improvement (%)": values.get("improvement_percentage", 0)
                    }
                    
                    comm_rows.append(row)
        
        if not comm_rows:
            logging.warning("No communication efficiency data found for LaTeX table")
            return
        
        # Create DataFrame
        df = pd.DataFrame(comm_rows)
        
        # Sort by dataset, alpha, algorithm
        df = df.sort_values(by=["Dataset", "Alpha", "Algorithm"])
        
        # Generate LaTeX table
        caption = "Communication Efficiency Comparison"
        label = "communication_efficiency"
        latex_table = dataframe_to_latex(df, caption, label, float_format="{:.2f}")
        
        # Save to file
        latex_file = os.path.join(latex_dir, "communication_efficiency.tex")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        logging.info(f"Generated communication efficiency LaTeX table: {latex_file}")
    except Exception as e:
        logging.error(f"Error generating communication efficiency LaTeX table: {str(e)}")

def main():
    """Main function to run the analysis."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze LCN experiment results")
    parser.add_argument("--results_dir", type=str, default="lcn_results",
                        help="Directory containing the results")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting analysis of LCN experiment results")
    
    # Create output directories
    plots_dir, latex_dir = create_output_dirs(args.results_dir)
    
    # Load experiment data
    logger.info(f"Loading experiment data from {args.results_dir}...")
    data = load_experiment_data(args.results_dir)
    
    if not data["raw_data"] and not data["metrics"]:
        logger.error("No data found to analyze. Make sure the experiment has been run.")
        return
    
    # Generate plots
    generate_accuracy_plots(data, plots_dir)
    generate_learning_curves(data, plots_dir)
    generate_time_to_accuracy_plots(data, plots_dir)
    generate_improvement_heatmap(data, plots_dir)
    generate_algorithm_comparison_plot(data, plots_dir)
    
    # Generate LaTeX tables
    generate_latex_tables(data, latex_dir)
    
    logger.info("Analysis completed successfully")
    print(f"Plots saved to {plots_dir}")
    print(f"LaTeX tables saved to {latex_dir}")

if __name__ == "__main__":
    main()
