import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Distiller:
    def __init__(self, student_model, temperature=2.0):
        self.student_model = student_model
        self.temperature = temperature
        self.device = next(student_model.parameters()).device
        self.student_model.to(self.device)

    def distillation_loss(self, student_logits, soft_labels, alpha=0.1):
        """
        Compute the distillation loss.
        student_logits and soft_labels should be raw logits, not probabilities.
        """
        T = self.temperature
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(soft_labels / T, dim=1)
        ) * (T * T)
        hard_loss = F.cross_entropy(student_logits, soft_labels.argmax(dim=1))
        return alpha * soft_loss + (1 - alpha) * hard_loss

    def train_student_model(self, synthetic_data, num_epochs=50, batch_size=64, learning_rate=0.001):
        inputs = torch.stack([item[0] for item in synthetic_data]).to(self.device)
        soft_labels = torch.stack([item[1] for item in synthetic_data]).to(self.device)
        dataset = TensorDataset(inputs, soft_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        self.student_model.train()
        
        loss_history = []
        accuracy_history = []

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch_inputs, batch_soft_labels in dataloader:
                optimizer.zero_grad()
                
                student_logits = self.student_model(batch_inputs)
                loss = self.distillation_loss(student_logits, batch_soft_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(student_logits.data, 1)
                total += batch_soft_labels.size(0)
                correct += (predicted == torch.argmax(batch_soft_labels, dim=1)).sum().item()

            epoch_loss = total_loss / len(dataloader)
            epoch_accuracy = correct / total
            loss_history.append(epoch_loss)
            accuracy_history.append(epoch_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            # Print some sample predictions for debugging
            if epoch % 10 == 0:
                with torch.no_grad():
                    sample_input = batch_inputs[:5]
                    sample_logits = self.student_model(sample_input)
                    sample_preds = F.softmax(sample_logits, dim=1)
                    print("Sample predictions:")
                    print(sample_preds)
                    print("Corresponding soft labels:")
                    print(F.softmax(batch_soft_labels[:5], dim=1))

            scheduler.step()

        return self.student_model, loss_history, accuracy_history

    def evaluate_model(self, model, test_data):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        all_predictions = []
        all_labels = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / len(test_data)

        return accuracy, avg_loss, all_predictions, all_labels