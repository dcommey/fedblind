import torch
import torch.nn as nn
import logging

class HARTeacher(nn.Module):
    def __init__(self, input_size=561, hidden_size=100, num_classes=6):
        """
        Human Activity Recognition (HAR) model.
        The HAR dataset has 6 classes (0-5 after zero-indexing).
        """
        super(HARTeacher, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def forward(self, x):
        # Log input shape for debugging
        original_shape = x.shape
        
        # Handle different input shapes
        if len(x.shape) == 1:  # Single sample without batch dimension
            x = x.unsqueeze(0)  # Add batch dimension
            
        if len(x.shape) > 2:  # If input has more dimensions than expected
            # Reshape to (batch_size, flattened_features)
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
            
            # If reshaped input size doesn't match expected input_size, truncate or pad
            if x.shape[1] != self.input_size:
                if x.shape[1] > self.input_size:
                    # Truncate to expected input size
                    self.logger.warning(f"Truncating input from {x.shape[1]} to {self.input_size} features")
                    x = x[:, :self.input_size]
                else:
                    # Pad with zeros to match expected input size
                    self.logger.warning(f"Padding input from {x.shape[1]} to {self.input_size} features")
                    padding = torch.zeros(batch_size, self.input_size - x.shape[1], device=x.device)
                    x = torch.cat([x, padding], dim=1)
        
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class HARStudent(nn.Module):
    def __init__(self, input_size=561, num_classes=6):
        """
        A simpler student model for HAR with fewer parameters.
        """
        super(HARStudent, self).__init__()
        self.input_size = input_size
        # Simplify the architecture for HAR dataset
        self.fc1 = nn.Linear(input_size, 100)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 50) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, num_classes)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def forward(self, x):
        # Reshape handling
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # If we get a multidimensional input, flatten it to match our expected input size
        if len(x.shape) > 2:
            batch_size = x.size(0)
            # Fully flatten to 2D: [batch_size, features]
            x = x.view(batch_size, -1)
        
        # Check dimensions and adapt if needed
        if x.shape[1] != self.input_size:
            batch_size = x.shape[0]
            if x.shape[1] > self.input_size:
                # Truncate to expected input size
                self.logger.warning(f"Truncating input from {x.shape[1]} to {self.input_size}")
                x = x[:, :self.input_size]
            else:
                # Pad with zeros
                self.logger.warning(f"Padding input from {x.shape[1]} to {self.input_size}")
                padding = torch.zeros(batch_size, self.input_size - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Forward pass with fixed architecture
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x