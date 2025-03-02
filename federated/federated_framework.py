# federated/federated_framework.py

import logging
from typing import List, Tuple, Optional
import numpy as np
import os
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
        
        # Use getattr with default values for attributes that might not exist
        self.clustering = getattr(config, 'clustering', None)  
        self.use_clustering = getattr(config, 'use_clustering', False)
        self.TeacherModel = getattr(config, 'TeacherModel', None)
        
        if not getattr(config, 'model_class', None):
            config.model_class = self.TeacherModel  # Fallback if not set
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cluster_assignments = None
        self.cluster_models = []
        self.target_accuracy = getattr(config, 'target_accuracy', None)
        self.rounds_to_target = None
        self.temperature = getattr(config, 'distillation_temperature', 2.0)
        
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
        if not self.use_clustering or not self.clustering:
            # Initialize with default single cluster
            self.cluster_assignments = [0] * len(self.clients)
            self.client_clusters = [self.clients]
            self.server.client_clusters = self.client_clusters
            return

        # If clustering is provided, use it
        if self.clustering is not None:
            try:
                cluster_results = self.clustering.cluster_clients(self.clients)
                if isinstance(cluster_results, tuple):
                    self.cluster_assignments, self.clustering_quality_metrics = cluster_results
                else:
                    self.cluster_assignments = cluster_results
                    self.clustering_quality_metrics = {}
            except Exception as e:
                self.logger.error(f"Error during clustering: {e}")
                # Fallback to single cluster on error
                self.cluster_assignments = [0] * len(self.clients)
                self.clustering_quality_metrics = {"error": str(e)}

        # If no clustering performed, fallback to single cluster
        if not hasattr(self, 'cluster_assignments') or self.cluster_assignments is None:
            self.cluster_assignments = [0] * len(self.clients)

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

    def train(self) -> None:
        """Run the federated learning training process."""
        num_clusters = len(set(self.cluster_assignments))
        self.logger.info(f"Starting training with {num_clusters} clusters")
        
        # Store cluster assignments for clarity
        cluster_to_clients = {}
        for cluster_id in range(num_clusters):
            # Get the actual client objects for this cluster
            cluster_to_clients[cluster_id] = [
                client for i, client in enumerate(self.clients) 
                if self.cluster_assignments[i] == cluster_id
            ]
            self.logger.info(f"Cluster {cluster_id} has {len(cluster_to_clients[cluster_id])} clients")
        
        # Train each cluster separately
        for cluster_id in range(num_clusters):
            self.logger.info(f"Training cluster {cluster_id + 1}/{num_clusters}")
            clients_in_cluster = cluster_to_clients[cluster_id]
            self.logger.info(f"Cluster {cluster_id + 1} has {len(clients_in_cluster)} clients")
            
            # Initialize a new global model for this cluster
            cluster_model = self.TeacherModel().to(self.device)
            
            # Set the server's clients to only those in this cluster
            self.server.clients = clients_in_cluster
            
            # Set the global model in the server
            self.server.global_model = cluster_model
            
            cluster_loss_history = []
            cluster_accuracy_history = []

            for round in range(self.config.num_rounds):
                self.logger.info(f"Starting round {round + 1}/{self.config.num_rounds} for cluster {cluster_id + 1}")
                
                # Train the cluster model
                self.server.train_round(round)
                
                # Evaluate the model
                metrics = self.server.evaluate_global_model()
                
                if metrics:
                    cluster_loss_history.append(metrics.get('loss', 0))
                    cluster_accuracy_history.append(metrics.get('accuracy', 0))
                    
                    self.logger.info(f"Round {round + 1} Metrics - Loss: {metrics.get('loss', 0):.4f}, "
                                f"Accuracy: {metrics.get('accuracy', 0):.4f}")
                    
                    # Check if target accuracy is reached
                    if self.target_accuracy and metrics['accuracy'] >= self.target_accuracy:
                        self.logger.info(f"Target accuracy {self.target_accuracy} reached in round {round + 1}")
                        if self.rounds_to_target is None:
                            self.rounds_to_target = round + 1
                        break

            # Store the trained model for this cluster
            self.cluster_models.append(self.server.global_model)
            self.loss_history[cluster_id] = cluster_loss_history
            self.accuracy_history[cluster_id] = cluster_accuracy_history
            self.logger.info(f"Finished training for cluster {cluster_id + 1}/{num_clusters}")

        # Final summary
        self.logger.info("\nTraining Summary:")
        self.logger.info(f"Total clusters: {num_clusters}")
        self.logger.info(f"Rounds per cluster: {self.config.num_rounds}")
        self.logger.info(f"Total cluster models: {len(self.cluster_models)}")

        # Store the final results including cluster information
        final_results = {
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'cluster_assignments': self.cluster_assignments,
            'num_clusters': num_clusters,
            'cluster_sizes': [self.cluster_assignments.count(i) for i in range(num_clusters)]
        }
        
        # Perform distillation if enabled
        if self.use_clustering and getattr(self.config, 'use_knowledge_distillation', False):
            try:
                unlabeled_data = self.get_unlabeled_data()
                if unlabeled_data is None:
                    self.logger.warning("Skipping distillation due to missing unlabeled data")
                    return final_results
                    
                distillation_results = self.final_distillation(unlabeled_data)
                self.logger.info("\nFinal Student Model Results:")
                self.logger.info(f"Accuracy: {distillation_results['accuracy']:.4f}")
                self.logger.info(f"Loss: {distillation_results['loss']:.4f}")
                self.logger.info(f"F1 Score: {distillation_results['f1_score']:.4f}")
                
                # Combine cluster and distillation results
                final_results.update({
                    'final_accuracy': distillation_results['accuracy'],
                    'final_loss': distillation_results['loss'],
                    'final_f1_score': distillation_results['f1_score'],
                    'distillation_results': distillation_results
                })
            except Exception as e:
                self.logger.error(f"Error in distillation process: {str(e)}")
                # Continue with partial results
            
            return final_results
        else:
            # If not using distillation, use the last cluster's final metrics
            last_cluster_id = num_clusters - 1
            if self.accuracy_history and last_cluster_id in self.accuracy_history:
                cluster_accuracy = self.accuracy_history[last_cluster_id][-1] if self.accuracy_history[last_cluster_id] else None
                cluster_loss = self.loss_history[last_cluster_id][-1] if self.loss_history[last_cluster_id] else None
                
                final_results.update({
                    'final_accuracy': cluster_accuracy,
                    'final_loss': cluster_loss
                })
            
            return final_results

    def get_student_model_results(self):
        """Get the results from the server."""
        return self.server.get_results()

    def get_unlabeled_data(self):
        """Retrieve the unlabeled data for final distillation."""
        if not self.unlabeled_loader:
            self.logger.warning("No unlabeled loader provided")
            return None
            
        try:
            unlabeled_data = []
            for batch in self.unlabeled_loader:
                # Handle different return types from the unlabeled loader
                if isinstance(batch, torch.Tensor):
                    # Direct tensor (for HAR dataset)
                    unlabeled_data.append(batch)
                elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                    # First item is the data (standard pattern)
                    unlabeled_data.append(batch[0])
                    
            if not unlabeled_data:
                self.logger.error("No unlabeled data could be loaded")
                return None
                
            return torch.cat(unlabeled_data, dim=0)
        except Exception as e:
            self.logger.error(f"Error loading unlabeled data: {str(e)}")
            return None

    def final_distillation(self, unlabeled_data):
        """
        Perform knowledge distillation using cluster models as teachers.
        """
        self.logger.info("Starting final knowledge distillation process")
        
        # Ensure proper input dimensions
        try:
            # Initialize student model with proper input size
            student_model = self._initialize_student_model()
            student_model = student_model.to(self.device)

            # Get teacher predictions (raw logits) with dimension validation
            unlabeled_data = unlabeled_data.to(self.device)
            teacher_logits = self._get_teacher_predictions(unlabeled_data)
            
            self.logger.info(f"Teacher logits shape: {teacher_logits.shape}")
            self.logger.info(f"Student model expects input shape matching: [{unlabeled_data.shape}]")
            
            # Create distillation dataset
            distill_dataset = TensorDataset(unlabeled_data, teacher_logits)
            
            # Also use some labeled data for direct supervision
            labeled_samples = []
            labeled_targets = []
            for batch in self.labeled_subset_loader:
                inputs, targets = batch
                labeled_samples.append(inputs)
                labeled_targets.append(targets)
            
            labeled_data = torch.cat(labeled_samples, dim=0)
            labeled_targets = torch.cat(labeled_targets, dim=0)
            
            # Convert hard targets to one-hot encoding for consistent format
            num_classes = teacher_logits.size(1)
            one_hot_targets = F.one_hot(labeled_targets, num_classes=num_classes).float()
            
            # Create dataset with labeled data
            labeled_dataset = TensorDataset(labeled_data, one_hot_targets)
            
            # Combine distillation and direct supervision datasets
            combined_dataset = torch.utils.data.ConcatDataset([distill_dataset, labeled_dataset])
            
            # Create final training dataloader
            train_loader = DataLoader(combined_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = self.test_loader

            # Fix and improve distillation implementation
            return self._improved_distillation(student_model, train_loader, val_loader)
        except Exception as e:
            self.logger.error(f"Error setting up distillation: {str(e)}")
            raise
        
    def _improved_distillation(self, student_model, train_loader, val_loader):
        """Improved distillation implementation."""
        # Setup training
        optimizer = torch.optim.Adam(
            student_model.parameters(), 
            lr=self.config.distillation_learning_rate, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training loop setup
        best_val_loss = float('inf')
        patience = 20
        no_improve = 0
        train_loss_history = []
        val_loss_history = []
        val_accuracy_history = []

        # Training loop
        for epoch in range(self.config.distillation_epochs):
            # Training phase
            student_model.train()
            train_loss = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                student_logits = student_model(inputs)
                
                # For distillation data (soft targets)
                if targets.dim() == 2:
                    # Apply temperature scaling and calculate distillation loss
                    loss = self._improved_distillation_loss(student_logits, targets)
                else:
                    # For regular supervised learning with hard labels
                    loss = F.cross_entropy(student_logits, targets)
                
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_loss_history.append(train_loss)

            # Validation phase
            student_model.eval()
            val_loss, val_accuracy = self._validate_epoch(student_model, val_loader)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

            self.logger.info(f"Epoch {epoch+1}/{self.config.distillation_epochs}, "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Val Accuracy: {val_accuracy:.4f}")
            
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(student_model.state_dict(), 'best_student_model.pth')
            else:
                no_improve += 1
                if no_improve == patience:
                    self.logger.info("Early stopping triggered")
                    break

        # Load best model with weights_only=True to avoid the warning
        student_model.load_state_dict(torch.load('best_student_model.pth', weights_only=True))
        
        # Final evaluation
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
        
        self.logger.info(f"Final student model accuracy: {test_accuracy:.4f}")

        return results

    def _improved_distillation_loss(self, student_logits, teacher_logits):
        """
        Improved knowledge distillation loss combining soft and hard targets.
        
        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
        
        Returns:
            Combined distillation loss
        """
        # Temperature scaling
        T = self.config.distillation_temperature
        
        # Soft targets (KL divergence loss)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
        
        # Hard targets (cross entropy with argmax labels)
        hard_targets = torch.argmax(teacher_logits, dim=1)
        hard_loss = F.cross_entropy(student_logits, hard_targets)
        
        # Balance between soft and hard loss (favor soft targets)
        alpha = 0.7  # 70% soft targets, 30% hard targets
        return alpha * soft_loss + (1 - alpha) * hard_loss

    def _initialize_student_model(self):
        """Initialize an appropriate student model based on the dataset."""
        try:
            # Get feature count for HAR from the teacher model if possible
            input_size = None
            if hasattr(self, 'TeacherModel') and hasattr(self.TeacherModel(), 'input_size'):
                input_size = self.TeacherModel().input_size
                self.logger.info(f"Using teacher's input size: {input_size}")
            
            if self.config.dataset_name == 'mnist':
                return MNISTStudent()
            elif self.config.dataset_name == 'har':
                # Get the actual input size from our dataset or trained teacher models
                if input_size is None:
                    # Try to infer from cluster models
                    if self.cluster_models and hasattr(self.cluster_models[0], 'input_size'):
                        input_size = self.cluster_models[0].input_size
                        self.logger.info(f"Using cluster model's input size: {input_size}")
                    else:
                        # Default
                        input_size = 561
                
                self.logger.info(f"Creating HAR student model with input size: {input_size}")
                return HARStudent(input_size=input_size, num_classes=6)
            elif self.config.dataset_name == 'cifar10':
                return CIFAR10Student()
            elif self.config.dataset_name == 'svhn':
                return SVHNStudent()
            elif self.config.dataset_name == 'cifar100':
                return CIFAR100Student()
            else:
                raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        except Exception as e:
            self.logger.error(f"Error initializing student model: {str(e)}")
            raise

    def _get_teacher_predictions(self, unlabeled_data):
        """
        Get ensemble predictions from all teacher models.
        
        Args:
            unlabeled_data: Tensor of unlabeled input data
            
        Returns:
            Raw logits from ensemble of teachers
        """
        if not self.cluster_models:
            self.logger.error("No teacher models available for distillation")
            raise ValueError("No teacher models available")
            
        all_logits = []
        for i, teacher_model in enumerate(self.cluster_models):
            # Ensure the model is on the right device
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()
            
            with torch.no_grad():
                try:
                    # Get raw logits (not probabilities)
                    logits = teacher_model(unlabeled_data)
                    all_logits.append(logits)
                    self.logger.info(f"Teacher model {i} produced predictions successfully")
                except Exception as e:
                    self.logger.error(f"Error getting predictions from teacher {i}: {str(e)}")
                    continue
                    
            # Move back to CPU to save memory
            teacher_model = teacher_model.to('cpu')
        
        if not all_logits:
            raise ValueError("No valid predictions from any teacher model")
        
        # Average the raw logits from all teachers
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        return avg_logits

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
        if teacher_logits_or_labels.dim() == 1:  # It's hard labels
            return F.cross_entropy(student_logits, teacher_logits_or_labels)
        else:  # It's soft labels
            T = self.config.distillation_temperature
            soft_targets = F.softmax(teacher_logits_or_labels / T, dim=1)
            student_log_softmax = F.log_softmax(student_logits / T, dim=1)
            kl_div = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean')
            return kl_div * (T * T)

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
            "clustering_algorithm": type(self.clustering).__name__
        }

    def get_privacy_info(self) -> dict:
        """Get information about the privacy settings and usage."""
        return {"use_dp": False}

# End of federated_framework.py