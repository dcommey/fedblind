# baselines/fednova.py

import torch
from federated.base_server import BaseServer
from federated.base_client import BaseClient
from torch import nn, optim
import copy
import logging

class FedNovaServer(BaseServer):
    def __init__(self, config):
        super().__init__(config)
        self.global_model = config.model_class()
        self.logger = logging.getLogger(self.__class__.__name__)

    def train_round(self, round_number):
        """Execute one round of federated learning with FedNova."""
        client_updates = []
        normalization_factors = []
        
        for client in self.clients:
            # Each client trains with FedNova
            local_state_dict, loss, accuracy, norm_factor = client.train_nova_local_model(
                model=copy.deepcopy(self.global_model),
                num_epochs=self.config.local_epochs,
                batch_size=self.config.batch_size
            )
            client_updates.append(local_state_dict)
            normalization_factors.append(norm_factor)
            self.logger.info(f"Client {client.client_id} - Round {round_number} - "
                           f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                           f"Norm Factor: {norm_factor:.4f}")

        # Aggregate with FedNova normalization
        self.global_model = self.aggregate_models_fednova(client_updates, normalization_factors)
        
        # Evaluate global model
        if hasattr(self, 'test_loader'):
            metrics = self.evaluate_model(self.global_model, self.test_loader)
            self.logger.info(f"Round {round_number} - Global Model Accuracy: {metrics['accuracy']:.4f}")

    def get_results(self):
        """Return the training results."""
        return {
            "accuracy_history": self.accuracy_history,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else None
        }

    def aggregate_models_fednova(self, models, normalization_factors):
        """Aggregate with FedNova normalization."""
        if not models:
            return self.global_model

        # Calculate the sum of normalization factors
        total_norm = sum(normalization_factors)
        if total_norm == 0:
            return self.global_model

        # Initialize the aggregated state dict
        agg_state_dict = self.global_model.state_dict()

        for key in agg_state_dict.keys():
            stacked_params = torch.stack([model[key].float() for model in models])
            # Reshape normalization factors to match parameter dimensions
            norm_factors = torch.tensor(normalization_factors, device=stacked_params.device)
            norm_factors = norm_factors.view(-1, *([1] * (len(stacked_params.shape) - 1)))
            
            # Multiply each client's parameters by its normalization factor
            weighted_params = stacked_params * norm_factors
            
            # Sum the weighted parameters and normalize
            agg_state_dict[key] = weighted_params.sum(dim=0) / total_norm

        self.global_model.load_state_dict(agg_state_dict)
        return self.global_model

    def evaluate_global_model(self, test_data=None, test_labels=None):
        """Evaluate the global model on the test dataset.
        
        If test_data and test_labels are not provided, uses the server's test_loader instead.
        """
        if test_data is None and self.test_loader is not None:
            return self.evaluate_model(self.global_model, self.test_loader)
        elif test_data is not None:
            return self.evaluate_model(self.global_model, test_data, test_labels)
        else:
            self.logger.warning("No test data available for evaluation")
            return None

class FedNovaClient(BaseClient):
    def __init__(self, client_id, dataloader):
        self.client_id = client_id
        self.dataloader = dataloader
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dataloader(self, batch_size):
        """Return the client's dataloader."""
        return self.dataloader

    def train_nova_local_model(self, model, num_epochs, batch_size):
        """Train with FedNova's normalized updates."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        lr = 0.01  # Default learning rate if config not available
        if hasattr(self, 'config') and hasattr(self.config, 'learning_rate'):
            lr = self.config.learning_rate
        optimizer = optim.SGD(model.parameters(), lr=lr)

        dataloader = self.get_dataloader(batch_size)
        total_loss = 0
        correct = 0
        total = 0
        total_norm = 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                # Calculate gradient norm for normalization factor
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += torch.sum(param.grad.data ** 2).item()
                total_norm += grad_norm ** 0.5  # L2 norm

                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            total_loss += epoch_loss

        avg_loss = total_loss / (len(dataloader) * num_epochs)
        accuracy = correct / total
        normalization_factor = total_norm / (num_epochs * len(dataloader))

        return model.state_dict(), avg_loss, accuracy, normalization_factor
