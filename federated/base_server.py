# federated/base_server.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from opacus.grad_sample import GradSampleModule
import collections
import logging

class BaseServer:
    def __init__(self, config):
        self.config = config
        self.clients = []
        self.client_clusters = None
        self.global_model = None
        self.teacher_models = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.accuracy_history = []
        self.test_loader = None

    def initialize_clients(self, client_data, client_labels):
        from .base_client import BaseClient  # Import here to avoid circular import
        self.clients = [
            BaseClient(i, data, labels)
            for i, (data, labels) in enumerate(zip(client_data, client_labels))
        ]

    def aggregate_models(self, models):
        """Aggregate a list of model state_dicts by averaging."""
        if not models:
            return self.global_model

        aggregated_state = self.global_model.state_dict()
        for key in aggregated_state.keys():
            stacked = torch.stack([m[key].float() for m in models])
            aggregated_state[key] = torch.mean(stacked, dim=0)

        self.global_model.load_state_dict(aggregated_state)
        return self.global_model

    def evaluate_model(self, model, test_data, test_labels=None):
        """Evaluate the model on test data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # If test_data is a DataLoader, use it directly
        if isinstance(test_data, DataLoader):
            test_loader = test_data
        else:
            # Create a DataLoader from the provided data
            if test_labels is None:
                raise ValueError("test_labels must be provided when test_data is not a DataLoader")
            test_dataset = TensorDataset(
                test_data.clone().detach() if isinstance(test_data, torch.Tensor) else torch.tensor(test_data),
                test_labels.clone().detach() if isinstance(test_labels, torch.Tensor) else torch.tensor(test_labels)
            )
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        
        # Store accuracy in history
        self.accuracy_history.append(accuracy)
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }

    def evaluate_global_model(self, test_data=None, test_labels=None):
        """Evaluate the global model."""
        if test_data is None and self.test_loader is not None:
            return self.evaluate_model(self.global_model, self.test_loader)
        elif test_data is not None:
            return self.evaluate_model(self.global_model, test_data, test_labels)
        else:
            self.logger.warning("No test data available for evaluation")
            return None

    def get_results(self):
        """Return the training results."""
        return {
            "accuracy_history": self.accuracy_history,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else None
        }