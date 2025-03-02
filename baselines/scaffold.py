# baselines/scaffold.py

import torch
from federated.base_server import BaseServer
from federated.base_client import BaseClient
from torch import nn, optim
import copy
import logging

class ScaffoldClient(BaseClient):
    def __init__(self, client_id, dataloader):
        self.client_id = client_id
        self.dataloader = dataloader
        self.control_variate = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataloader(self, batch_size):
        """Return the client's dataloader."""
        return self.dataloader

    def train_scaffold_local_model(self, model, server_control_variate, num_epochs, batch_size):
        """Train with SCAFFOLD's control variates."""
        model = model.to(self.device)
        model.train()

        # Initialize client control variate if needed
        if self.control_variate is None:
            self.control_variate = [torch.zeros_like(param).to(self.device) 
                                  for param in model.parameters()]

        # Move server control variate to the same device
        server_control_variate = [sc.to(self.device) for sc in server_control_variate]

        criterion = nn.CrossEntropyLoss()
        lr = 0.01  # Default learning rate if config not available
        if hasattr(self, 'config') and hasattr(self.config, 'learning_rate'):
            lr = self.config.learning_rate

        # Store initial parameters
        initial_params = [param.clone() for param in model.parameters()]
        
        optimizer = optim.SGD(model.parameters(), lr=lr)
        dataloader = self.get_dataloader(batch_size)
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Update with control variate correction
                for param, c, sc in zip(model.parameters(), self.control_variate, server_control_variate):
                    if param.grad is not None:
                        param.grad = param.grad - c + sc  # All tensors should be on the same device now
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            total_loss += epoch_loss

        # Update client control variate
        new_control_variate = []
        steps = num_epochs * len(dataloader)
        for init_param, param, c, sc in zip(initial_params, model.parameters(), 
                                          self.control_variate, server_control_variate):
            update = (init_param - param) / (steps * lr)
            new_c = c + update - sc  # All tensors should be on the same device
            new_control_variate.append(new_c)

        self.control_variate = new_control_variate
        avg_loss = total_loss / (len(dataloader) * num_epochs)
        accuracy = correct / total

        return model.state_dict(), self.control_variate, avg_loss, accuracy

    def train_local_model(self, model, num_epochs, batch_size, learning_rate=0.01):
        """Override BaseClient's train_local_model to use SCAFFOLD-specific training."""
        # Initialize control variates if needed
        if self.control_variate is None:
            self.control_variate = [torch.zeros_like(param) for param in model.parameters()]
        
        # Get server's control variate from model's server_control attribute
        server_control_variate = getattr(model, 'server_control', None)
        if server_control_variate is None:
            # Initialize with zeros if not set
            server_control_variate = [torch.zeros_like(param) for param in model.parameters()]
            
        # Call SCAFFOLD-specific training method
        return self.train_scaffold_local_model(model, server_control_variate, num_epochs, batch_size)

    def get_control_variate(self):
        """Retrieve the client-side control variate."""
        return self.control_variate

    def set_updated_control_variate(self, updated_cv):
        """Update the client-side control variate."""
        self.control_variate = updated_cv

class ScaffoldServer(BaseServer):
    def __init__(self, config):
        super().__init__(config)
        self.global_model = config.model_class()
        self.server_control_variate = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_server_control_variate(self):
        """Initialize server control variate if not already initialized."""
        if self.server_control_variate is None:
            self.server_control_variate = [torch.zeros_like(param).to(self.device) 
                                         for param in self.global_model.parameters()]

    def train_round(self, round_number):
        """Execute one round of federated learning with SCAFFOLD."""
        self.initialize_server_control_variate()
        
        # Ensure global model is on the correct device
        self.global_model = self.global_model.to(self.device)
        
        client_updates = []
        cv_updates = []
        
        for client in self.clients:
            local_state_dict, client_cv, loss, accuracy = client.train_scaffold_local_model(
                model=copy.deepcopy(self.global_model),
                server_control_variate=self.server_control_variate,
                num_epochs=self.config.local_epochs,
                batch_size=self.config.batch_size
            )
            client_updates.append(local_state_dict)
            cv_updates.append(client_cv)
            
            self.logger.info(f"Client {client.client_id} - Round {round_number} - "
                           f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Aggregate model updates
        self.global_model = self.aggregate_models(client_updates)
        
        # Update server control variate
        if cv_updates:
            for i in range(len(self.server_control_variate)):
                cv_update = torch.stack([cv[i] for cv in cv_updates]).mean(dim=0)
                self.server_control_variate[i] = cv_update.to(self.device)
        
        # Evaluate global model
        if self.test_loader is not None:
            metrics = self.evaluate_global_model()
            self.logger.info(f"Round {round_number} - Global Model Accuracy: {metrics['accuracy']:.4f}")

    def evaluate_global_model(self, test_data=None, test_labels=None):
        """Evaluate the global model on test data."""
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
