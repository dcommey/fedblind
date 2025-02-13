# federated/base_client.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import logging

class BaseClient:
    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
        self.dataset = TensorDataset(self.data, self.labels)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train_local_model(self, model, num_epochs, batch_size, learning_rate):
        """Train the model locally on client data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = self.get_dataloader(batch_size)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader)
            self.logger.info(f"Client {self.client_id} - Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        return model

    def evaluate_model(self, model, test_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        test_loader = DataLoader(TensorDataset(torch.tensor(test_data[0]), torch.tensor(test_data[1])), batch_size=64)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy