import torch.nn as nn
import torch.nn.functional as F

class HARTeacher(nn.Module):
    def __init__(self):
        super(HARTeacher, self).__init__()
        self.fc1 = nn.Linear(561, 256)  # HAR dataset has 561 features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)    # 6 activity classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x 