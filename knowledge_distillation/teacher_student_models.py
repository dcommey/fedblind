# knowledge_distillation/teacher_student_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class MNISTTeacher(nn.Module):
    def __init__(self):
        super(MNISTTeacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MNISTStudent(nn.Module):
    def __init__(self):
        super(MNISTStudent, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HARTeacher(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HARTeacher, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 116, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 64 * 116)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HARStudent(nn.Module):
    def __init__(self, input_size=561, num_classes=6):
        """
        Student model for Human Activity Recognition dataset
        """
        super(HARStudent, self).__init__()
        self.input_size = input_size
        # Simpler architecture than teacher
        self.fc1 = nn.Linear(input_size, 100)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 50) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, num_classes)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def forward(self, x):
        # Reshape handling for different input types
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Convert any n-dimensional input to 2D [batch_size, features]
        if len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
        
        # Handle dimension mismatches
        if x.shape[1] != self.input_size:
            batch_size = x.shape[0]
            if x.shape[1] > self.input_size:
                # Log and truncate
                x = x[:, :self.input_size]
            else:
                # Log and pad
                padding = torch.zeros(batch_size, self.input_size - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Forward pass
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class CIFAR10Teacher(nn.Module):
    def __init__(self):
        super(CIFAR10Teacher, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Student(nn.Module):
    def __init__(self):
        super(CIFAR10Student, self).__init__()
        # Upgrade the student model architecture to better match the teacher's capacity
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x

class SVHNTeacher(nn.Module):
    def __init__(self):
        super(SVHNTeacher, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SVHNStudent(nn.Module):
    def __init__(self):
        super(SVHNStudent, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR100Teacher(nn.Module):
    def __init__(self):
        super(CIFAR100Teacher, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR100Student(nn.Module):
    def __init__(self):
        super(CIFAR100Student, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
