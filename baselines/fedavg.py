# baselines/fedavg.py

import torch
from federated.base_server import BaseServer
from federated.base_client import BaseClient
import copy
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Added missing import
from torch.utils.data import DataLoader, TensorDataset

class FedAvgClient(BaseClient):
    def __init__(self, client_id, dataloader):
        self.client_id = client_id
        self.dataloader = dataloader
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dataloader(self, batch_size):
        """Return the client's dataloader."""
        return self.dataloader

    def train_local_model(self, model, num_epochs, batch_size):
        """Train the local model."""
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

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            total_loss += epoch_loss

        avg_loss = total_loss / (len(dataloader) * num_epochs)
        accuracy = correct / total

        return model.state_dict(), avg_loss, accuracy

class FedAvgServer(BaseServer):
    def __init__(self, config):
        super().__init__(config)
        self.global_model = config.model_class()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_round(self, round_number):
        """Execute one round of federated learning."""
        client_updates = []
        
        for client in self.clients:
            # Train each client with the current global model
            local_state_dict, loss, accuracy = client.train_local_model(
                model=copy.deepcopy(self.global_model),
                num_epochs=self.config.local_epochs,
                batch_size=self.config.batch_size
            )
            client_updates.append(local_state_dict)
            
            self.logger.info(f"Client {client.client_id} - Round {round_number} - "
                           f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Aggregate client updates
        self.global_model = self.aggregate_models(client_updates)
        
        # Evaluate global model
        if hasattr(self, 'test_loader'):
            metrics = self.evaluate_model(self.global_model, self.test_loader)
            self.logger.info(f"Round {round_number} - Global Model Accuracy: {metrics['accuracy']:.4f}")

    def aggregate_models(self, models):
        """Aggregate models by simple averaging."""
        if not models:
            return self.global_model

        # Initialize aggregated state dict
        agg_state_dict = self.global_model.state_dict()

        for key in agg_state_dict.keys():
            stacked_params = torch.stack([model[key] for model in models])

            # Average the parameters
            agg_state_dict[key] = stacked_params.mean(dim=0)

        # Load averaged parameters into the global model
        self.global_model.load_state_dict(agg_state_dict)
        return self.global_model

    def evaluate_global_model(self):
        """Evaluate the global model on the test dataset."""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions and labels for F1 score calculation
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / total
        
        # Calculate F1 score with zero_division=0 to avoid warnings
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        self.logger.info(f'Test set: Average loss: {test_loss:.4f}, '
                        f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }

    def evaluate_global_model(self, test_data=None, test_labels=None):
        """Evaluate the global model on test data.
        
        If test_data and test_labels are not provided, uses the server's test_loader instead.
        """
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