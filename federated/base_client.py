# federated/base_client.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

class BaseClient:
    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
        self.dataset = TensorDataset(self.data, self.labels)

    def get_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train_local_model(self, model, num_epochs, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        dataloader = self.get_dataloader(batch_size)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(num_epochs):
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return model.state_dict(), avg_loss, accuracy

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