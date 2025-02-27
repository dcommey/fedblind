# federated/federated_framework.py

import logging
from typing import List, Tuple, Optional
import numpy as np
import os
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from .base_server import BaseServer
from .base_client import BaseClient
from tqdm import tqdm
from knowledge_distillation.distiller import Distiller
from knowledge_distillation.teacher_student_models import (
    MNISTStudent, HARStudent, CIFAR10Student, SVHNStudent, CIFAR100Student
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FederatedLearningFramework:
    def __init__(self, config):
        self.config = config
        self.server = None
        self.clients = None
        self.test_loader = None
        self.unlabeled_loader = None
        self.labeled_subset_loader = None
        self.algorithm = config.algorithm
        self.logger = logging.getLogger(self.__class__.__name__)
        self.clustering = getattr(config, 'clustering', None)  # Use getattr with a default value
        self.use_clustering = getattr(config, 'use_clustering', False)
        self.TeacherModel = getattr(config, 'TeacherModel', None)
        
        if not getattr(config, 'model_class', None):
            config.model_class = self.TeacherModel  # Fallback if not set
            
        # Add accuracy threshold tracking
        self.accuracy_thresholds = getattr(config, 'accuracy_thresholds', [0.6, 0.7, 0.8])
        self.threshold_rounds = {threshold: None for threshold in self.accuracy_thresholds}
        self.communication_cost = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cluster_assignments = None
        self.cluster_models = []
        self.target_accuracy = config.target_accuracy
        self.rounds_to_target = None
        self.temperature = config.distillation_temperature
        
        self.loss_history = {} 
        self.accuracy_history = {}
        self.student_loss_history = []
        self.student_accuracy_history = []
        self.logger.info(f"Initializing framework with {self.algorithm} algorithm")

    def initialize(self, client_dataloaders: List[DataLoader], 
                  test_loader: DataLoader, 
                  unlabeled_loader: Optional[DataLoader] = None,
                  labeled_subset_loader: Optional[DataLoader] = None) -> None:
        """Initialize the federated learning framework."""
        self.server = self._create_server()
        
        # Create appropriate client instances based on algorithm
        if self.algorithm == 'fedprox':
            from baselines.fedprox import FedProxClient
            clients = [FedProxClient(i, dataloader, proximal_mu=self.config.proximal_mu) 
                      for i, dataloader in enumerate(client_dataloaders)]
        elif self.algorithm == 'scaffold':
            from baselines.scaffold import ScaffoldClient
            clients = [ScaffoldClient(i, dataloader) 
                      for i, dataloader in enumerate(client_dataloaders)]
        elif self.algorithm == 'fednova':
            from baselines.fednova import FedNovaClient
            clients = [FedNovaClient(i, dataloader) 
                      for i, dataloader in enumerate(client_dataloaders)]
        else:  # default to fedavg
            from baselines.fedavg import FedAvgClient
            clients = [FedAvgClient(i, dataloader) 
                      for i, dataloader in enumerate(client_dataloaders)]

        self.server.clients = clients
        self.clients = clients
        self.test_loader = test_loader
        self.unlabeled_loader = unlabeled_loader
        self.labeled_subset_loader = labeled_subset_loader
        
        # Set the test_loader in the server
        self.server.test_loader = test_loader
        
        self._perform_clustering()

    def _create_server(self):
        """Create the appropriate server instance based on the algorithm."""
        if self.algorithm == 'fedprox':
            from baselines.fedprox import FedProxServer
            return FedProxServer(self.config)
        elif self.algorithm == 'scaffold':
            from baselines.scaffold import ScaffoldServer
            return ScaffoldServer(self.config)
        elif self.algorithm == 'fednova':
            from baselines.fednova import FedNovaServer
            return FedNovaServer(self.config)
        else:  # default to fedavg
            from baselines.fedavg import FedAvgServer
            return FedAvgServer(self.config)

    def _perform_clustering(self):
        """Perform client clustering if specified in config."""
        if not hasattr(self.config, 'use_clustering') or not self.config.use_clustering:
            # Initialize with default single cluster
            self.cluster_assignments = [0] * len(self.clients)
            self.client_clusters = [self.clients]
            self.server.client_clusters = self.client_clusters
            return

        if hasattr(self.config, 'use_dp_clustering') and self.config.use_dp_clustering:
            from clustering.dp_quantile_clustering import DPQuantileClustering
            clustering = DPQuantileClustering(
                epsilon=self.config.cl_epsilon,
                num_quantiles=self.config.num_quantiles
            )
        else:
            from clustering.non_dp_quantile_clustering import NonDPQuantileClustering
            clustering = NonDPQuantileClustering(
                num_quantiles=self.config.num_quantiles
            )
        
        result = clustering.cluster_clients(self.clients)
        if isinstance(result, tuple):
            self.cluster_assignments, self.clustering_quality_metrics = result
        else:
            self.cluster_assignments = result
            self.clustering_quality_metrics = {}

        # Group clients by cluster
        unique_clusters = set(self.cluster_assignments)
        self.client_clusters = [
            [client for client, cluster_id in zip(self.clients, self.cluster_assignments) 
             if cluster_id == cluster] 
            for cluster in unique_clusters
        ]
        
        # Set clusters in server
        self.server.client_clusters = self.client_clusters
        
        self.logger.info(f"Created {len(self.client_clusters)} clusters")
        for i, cluster in enumerate(self.client_clusters):
            self.logger.info(f"Cluster {i} has {len(cluster)} clients")

    def get_student_model_results(self):
        """Get the final student model results from knowledge distillation if available."""
        return getattr(self, 'final_student_results', self.server.get_results())

    def get_unlabeled_data(self):
        """Retrieve the unlabeled data for final distillation."""
        unlabeled_data = []
        for batch in self.unlabeled_loader:
            unlabeled_data.append(batch)
        return torch.cat(unlabeled_data, dim=0)

    def train(self, progress_callback=None) -> None:
        """Run the federated learning training process."""
        if not self.server or not self.clients:
            raise ValueError("Framework not initialized. Call initialize() first.")

        num_clusters = len(set(self.cluster_assignments))
        self.logger.info(f"Starting training with {num_clusters} clusters")

        # Create clusters
        clusters = {}
        for cluster_id in range(num_clusters):
            cluster_clients = [
                self.clients[i] for i, assigned_cluster in enumerate(self.cluster_assignments) 
                if assigned_cluster == cluster_id
            ]
            clusters[cluster_id] = cluster_clients
            self.logger.info(f"Cluster {cluster_id} has {len(cluster_clients)} clients")
        
        # Train each cluster
        for cluster_id, cluster_clients in clusters.items():
            self.logger.info(f"\nTraining cluster {cluster_id + 1}/{num_clusters}")
            
            # Initialize history tracking for this cluster
            self.loss_history[cluster_id] = []
            self.accuracy_history[cluster_id] = []
            
            # Set the current cluster's clients in the server
            self.server.clients = cluster_clients
            
            # Initialize a new global model for this cluster
            self.server.global_model = self.config.model_class().to(self.device)
            
            # Train for specified number of rounds
            for round in range(self.config.num_rounds):
                # Train round
                self.server.train_round(round)
                
                # Evaluate after each round
                if self.test_loader is not None:
                    # Prepare test data
                    test_data = torch.cat([batch[0] for batch in self.test_loader], dim=0)
                    test_labels = torch.cat([batch[1] for batch in self.test_loader], dim=0)
                    
                    metrics = self.server.evaluate_global_model(test_data, test_labels)
                    if metrics:
                        # Store metrics
                        self.loss_history[cluster_id].append(metrics.get('loss', 0))
                        self.accuracy_history[cluster_id].append(metrics.get('accuracy', 0))
                        
                        # Report progress
                        if progress_callback:
                            progress_callback(round + 1, metrics)
                        
                        # Log progress
                    test_data = torch.cat([batch[0] for batch in self.test_loader], dim=0)
                    test_labels = torch.cat([batch[1] for batch in self.test_loader], dim=0)
                    
                    metrics = self.server.evaluate_global_model(test_data, test_labels)
                    if metrics:
                        # Store metrics
                        self.loss_history[cluster_id].append(metrics.get('loss', 0))
                        self.accuracy_history[cluster_id].append(metrics.get('accuracy', 0))
                        
                        # Report progress
                        if progress_callback:
                            progress_callback(round + 1, metrics)
                        
                        # Log progress
                        self.logger.info(f"Round {round + 1}/{self.config.num_rounds} - "
                                    f"Cluster {cluster_id + 1}/{num_clusters} - "
                                    f"Loss: {metrics.get('loss', 0):.4f}, "
                                    f"Accuracy: {metrics.get('accuracy', 0):.4f}")
                        
                        # Check if target accuracy is reached
                        if self.target_accuracy and metrics['accuracy'] >= self.target_accuracy:
                            self.logger.info(f"Target accuracy {self.target_accuracy} reached in round {round + 1}")
                            if self.rounds_to_target is None:
                                self.rounds_to_target = round + 1
                            break

                        # Check for accuracy thresholds
                        for threshold in self.accuracy_thresholds:
                            if metrics['accuracy'] >= threshold and self.threshold_rounds[threshold] is None:
                                self.threshold_rounds[threshold] = round + 1
                                self.logger.info(f"Reached {threshold} accuracy in round {round + 1}")

            # Store trained cluster model
            self.cluster_models.append(copy.deepcopy(self.server.global_model))
            
        # Final summary
        self.logger.info("\nTraining Summary:")
        self.logger.info(f"Total clusters: {num_clusters}")
        self.logger.info(f"Rounds per cluster: {self.config.num_rounds}")
        self.logger.info(f"Total cluster models: {len(self.cluster_models)}")
        
        # Print final accuracies for each cluster
        for cluster_id in range(num_clusters):
            final_accuracy = self.accuracy_history[cluster_id][-1] if self.accuracy_history[cluster_id] else 0
            self.logger.info(f"Cluster {cluster_id + 1} final accuracy: {final_accuracy:.4f}")

        # Perform knowledge distillation if specified
        if self.use_clustering and self.config.use_knowledge_distillation:
            self.logger.info("\nStarting Knowledge Distillation...")
            unlabeled_data = self.get_unlabeled_data()
            final_results = self.final_distillation(unlabeled_data)
            return final_results
        
        return None

    def final_distillation(self, unlabeled_data):
        student_model = self._initialize_student_model()
        student_model = student_model.to(self.device)

        unlabeled_data = unlabeled_data.to(self.device)
        teacher_predictions = self._get_teacher_predictions(unlabeled_data)

        # Use the labeled subset loader
        labeled_data = next(iter(self.labeled_subset_loader))
        labeled_inputs, labeled_targets = labeled_data[0].to(self.device), labeled_data[1].to(self.device)

        # Ensure teacher_predictions and labeled_targets have the same number of dimensions
        if teacher_predictions.dim() == 2 and labeled_targets.dim() == 1:
            labeled_targets = F.one_hot(labeled_targets, num_classes=teacher_predictions.size(1)).float()
        
        mixed_data = torch.cat([unlabeled_data, labeled_inputs], dim=0)
        mixed_targets = torch.cat([teacher_predictions, labeled_targets], dim=0)

        dataset = TensorDataset(mixed_data, mixed_targets)
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        val_loader = self.test_loader

        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.config.distillation_learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_val_loss = float('inf')
        patience = 20
        no_improve = 0
        train_loss_history = []
        val_loss_history = []
        val_accuracy_history = []

        for epoch in range(self.config.distillation_epochs):
            student_model.train()
            train_loss = self._train_epoch(student_model, train_loader, optimizer)
            train_loss_history.append(train_loss)

            student_model.eval()
            val_loss, val_accuracy = self._validate_epoch(student_model, val_loader)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

            print(f"Epoch {epoch+1}/{self.config.distillation_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(student_model.state_dict(), 'best_student_model.pth')
            else:
                no_improve += 1
                if no_improve == patience:
                    print("Early stopping triggered")
                    break

            if (epoch + 1) % 10 == 0:
                self._print_sample_predictions(student_model, val_loader)

        student_model.load_state_dict(torch.load('best_student_model.pth'))
        
        test_accuracy, test_loss, predictions, true_labels = self._evaluate_model(student_model, self.test_loader)

        results = self._compute_metrics(predictions, true_labels)
        results.update({
            'accuracy': test_accuracy,
            'loss': test_loss,
            'test_loss': test_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history
        })

        self.student_model = student_model
        self.final_student_results = results

        return results

    def _initialize_student_model(self):
        if self.config.dataset_name == 'mnist':
            return MNISTStudent()
        elif self.config.dataset_name == 'har':
            return HARStudent(input_size=9, num_classes=6)
        elif self.config.dataset_name == 'cifar10':
            return CIFAR10Student()
        elif self.config.dataset_name == 'svhn':
            return SVHNStudent()
        elif self.config.dataset_name == 'cifar100':
            return CIFAR100Student()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")

    def _get_teacher_predictions(self, unlabeled_data):
        all_predictions = []
        for teacher_model in self.cluster_models:
            teacher_model.to(self.device)
            teacher_model.eval()
            with torch.no_grad():
                predictions = teacher_model(unlabeled_data.to(self.device))
                # Ensure predictions are probabilities
                predictions = F.softmax(predictions, dim=1)
                all_predictions.append(predictions)
            teacher_model.to('cpu')
        # Average the predictions from all teachers
        return torch.mean(torch.stack(all_predictions), dim=0)

    def _train_epoch(self, model, dataloader, optimizer):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self._distillation_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _validate_epoch(self, model, dataloader):
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return total_loss / len(dataloader), correct / total

    def _evaluate_model(self, model, dataloader):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss, all_predictions, all_labels

    def _compute_metrics(self, predictions, true_labels):
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1_score': f1_score(true_labels, predictions, average='weighted')
        }

    def _print_sample_predictions(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            inputs, labels = next(iter(dataloader))
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print("\nSample predictions:")
            for i in range(5):
                print(f"True: {labels[i].item()}, Predicted: {predicted[i].item()}")

    def _distillation_loss(self, student_logits, teacher_logits_or_labels):
        """Improved distillation loss with balanced soft/hard components."""
        if teacher_logits_or_labels.dim() == 1:  # It's hard labels
            return F.cross_entropy(student_logits, teacher_logits_or_labels)
        else:  # It's soft labels (treated as scaled logits)
            # Balance between soft and hard losses
            alpha = 0.5  # Adjust this based on validation performance
            T = self.temperature
            
            # Soft loss with temperature scaling
            soft_targets = F.softmax(teacher_logits_or_labels / T, dim=1)
            student_log_softmax = F.log_softmax(student_logits / T, dim=1)
            soft_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean')
            soft_loss = soft_loss * (T * T)  # Important scaling factor
            
            # Hard loss (standard cross entropy with hard targets)
            hard_targets = torch.argmax(teacher_logits_or_labels, dim=1)
            hard_loss = F.cross_entropy(student_logits, hard_targets)
            
            # Combined loss
            return alpha * soft_loss + (1-alpha) * hard_loss

    def get_loss_history(self) -> dict:
        """Get the loss history for all clusters."""
        return self.loss_history
    
    def get_accuracy_history(self) -> dict:
        """Get the accuracy history for all clusters."""
        return self.accuracy_history
    
    def get_student_training_history(self):
        return {
            'loss_history': self.student_loss_history,
            'accuracy_history': self.student_accuracy_history
        }
    
    def get_cluster_assignments(self) -> List[int]:
        """Get the cluster assignments for all clients."""
        return self.cluster_assignments

    def get_cluster_models(self) -> List[torch.nn.Module]:
        """Get all trained cluster models (teacher models)."""
        return self.cluster_models

    def save_models(self, path: str) -> None:
        """Save all models (cluster models and student model) to the specified path."""
        os.makedirs(path, exist_ok=True)

        # Save cluster models
        if self.use_clustering:
            for i, model in enumerate(self.cluster_models):
                torch.save(model.state_dict(), os.path.join(path, f"cluster_model_{i}.pth"))
            self.logger.info(f"Saved {len(self.cluster_models)} cluster models to {path}")
        else:
            # Save the single model as "global_model.pth" when no clustering is used
            torch.save(self.cluster_models[0].state_dict(), os.path.join(path, "global_model.pth"))
            self.logger.info(f"Saved global model to {path}")

        # Save student model if it exists
        if hasattr(self, 'student_model'):
            torch.save(self.student_model.state_dict(), os.path.join(path, "student_model.pth"))
            self.logger.info(f"Saved student model to {path}")

    def load_models(self, path: str) -> None:
        """Load all cluster models from the specified path."""
        self.cluster_models = []
        i = 0
        while True:
            model_path = f"{path}/cluster_model_{i}.pth"
            if not os.path.exists(model_path):
                break
            model = self.TeacherModel()
            model.load_state_dict(torch.load(model_path))
            self.cluster_models.append(model)
            i += 1
        self.logger.info(f"Loaded {len(self.cluster_models)} cluster models from {path}")

    def get_config(self) -> dict:
        """Get the current configuration of the framework."""
        return vars(self.config)

    def update_config(self, **kwargs) -> None:
        """Update the configuration of the framework."""

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Config has no attribute '{key}', skipping.")
        self.logger.info("Updated framework configuration.")

    def get_clustering_info(self) -> dict:
        """Get information about the clustering process."""
        return {
            "num_clusters": len(set(self.cluster_assignments)),
            "cluster_sizes": [self.cluster_assignments.count(i) for i in range(len(set(self.cluster_assignments)))],
            "clustering_algorithm": self.clustering_quality_metrics.get('algorithm', 'Unknown'),
            "quality_metrics": self.clustering_quality_metrics
        }

    def get_privacy_info(self) -> dict:
        """Get information about the privacy settings and usage."""
        return {"use_dp": False}

    # Add method for tracking parameter size
    def calculate_model_size(self, model):
        """Calculate the size of a model in bytes"""
        size = 0
        for param in model.parameters():
            size += param.numel() * param.element_size()
    def get_privacy_info(self) -> dict:
        return size

# End of federated_framework.py
    def get_clustering_info(self) -> dict:
        """Get information about the clustering process."""
        return {
            "num_clusters": len(set(self.cluster_assignments)),
            "cluster_sizes": [self.cluster_assignments.count(i) for i in range(len(set(self.cluster_assignments)))],
            "clustering_algorithm": self.clustering_quality_metrics.get('algorithm', 'Unknown'),
            "quality_metrics": self.clustering_quality_metrics
        }

    def get_privacy_info(self) -> dict:
        """Get information about the privacy settings and usage."""
        return {"use_dp": False}

# End of federated_framework.py