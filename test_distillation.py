"""
Script to test knowledge distillation in isolation from the federated learning process.
This helps verify that the distillation process works correctly.
"""

import torch
import logging
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utils.config import Config
from knowledge_distillation.teacher_student_models import (
    MNISTTeacher, MNISTStudent,
    CIFAR10Teacher, CIFAR10Student
)
from data.data_loader import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def setup_config(dataset_name):
    """Create a configuration for testing distillation."""
    config = Config()
    config.dataset_name = dataset_name
    config.batch_size = 64
    config.distillation_temperature = 3.0  # Test with higher temperature
    config.distillation_learning_rate = 0.001
    config.distillation_epochs = 30
    config.unlabeled_ratio = 0.3
    config.labeled_subset_size = 1000
    config.num_clients = 3  # Use small number for quick testing
    
    # Set model class based on dataset
    if dataset_name == 'mnist':
        config.model_class = MNISTTeacher
        config.TeacherModel = MNISTTeacher
    elif dataset_name == 'cifar10':
        config.model_class = CIFAR10Teacher
        config.TeacherModel = CIFAR10Teacher
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for this test")
    
    return config

def create_teacher_models(dataset_name, num_teachers=2):
    """Create and initialize multiple teacher models."""
    if dataset_name == 'mnist':
        return [MNISTTeacher() for _ in range(num_teachers)]
    elif dataset_name == 'cifar10':
        return [CIFAR10Teacher() for _ in range(num_teachers)]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for this test")

def create_student_model(dataset_name):
    """Create an appropriate student model."""
    if dataset_name == 'mnist':
        return MNISTStudent()
    elif dataset_name == 'cifar10':
        return CIFAR10Student()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for this test")

def train_teacher_models(teachers, train_loader, num_epochs=5):
    """Train teacher models on different subsets of data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {len(teachers)} teacher models for {num_epochs} epochs each")
    
    for i, teacher in enumerate(teachers):
        teacher.to(device)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        logger.info(f"Training teacher {i+1}/{len(teachers)}")
        for epoch in range(num_epochs):
            teacher.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Each teacher sees different batches
                if batch_idx % len(teachers) != i:
                    continue
                    
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = teacher(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            accuracy = 100 * correct / total if total > 0 else 0
            logger.info(f"Teacher {i+1} - Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/total:.4f}, Accuracy: {accuracy:.2f}%")
    
    return teachers

def get_teacher_predictions(teachers, unlabeled_data, temperature=2.0):
    """Get ensemble predictions from multiple teacher models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlabeled_data = unlabeled_data.to(device)
    
    all_predictions = []
    for teacher in teachers:
        teacher.to(device)
        teacher.eval()
        with torch.no_grad():
            # Get logits directly (don't apply temperature yet)
            logits = teacher(unlabeled_data)
            # Store raw logits for later temperature scaling
            all_predictions.append(logits)
    
    # Average the logits from all teachers (ensemble)
    avg_logits = torch.mean(torch.stack(all_predictions), dim=0)
    
    # Return raw logits (not softmax probabilities)
    # We'll apply temperature in the loss function
    return avg_logits

def train_student_model(student, train_loader, val_loader, temperature=2.0, num_epochs=20):
    """Train student model using knowledge distillation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        student.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            student_logits = student(inputs)
            
            # For distillation data (soft targets)
            if targets.dim() == 2:
                # Temperature scaling for soft targets
                T = temperature
                # Convert teacher logits to probabilities with temperature
                teacher_probs = F.softmax(targets / T, dim=1)
                # Get student probabilities with temperature
                student_log_probs = F.log_softmax(student_logits / T, dim=1)
                # KL divergence loss
                loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                
                # Add hard label guidance using argmax of teacher logits
                hard_targets = torch.argmax(targets, dim=1)
                hard_loss = F.cross_entropy(student_logits, hard_targets)
                
                # Combined loss with weighted balance (favor soft targets)
                alpha = 0.7  # Higher weight for soft targets
                loss = alpha * loss + (1 - alpha) * hard_loss
            else:
                # For validation data (hard labels)
                loss = F.cross_entropy(student_logits, targets)
            
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        student.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                
                # For validation, we always use cross entropy with hard labels
                loss = F.cross_entropy(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student.state_dict(), 'best_student.pth', _use_new_zipfile_serialization=True)
    
    # Load best model
    student.load_state_dict(torch.load('best_student.pth', weights_only=True))
    
    return student, train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate model on test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def plot_learning_curves(train_losses, val_losses, val_accuracies):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('distillation_curves.png')
    logger.info("Learning curves saved as 'distillation_curves.png'")

def main():
    parser = argparse.ArgumentParser(description="Test knowledge distillation")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset to use (default: mnist)")
    parser.add_argument("--num_teachers", type=int, default=3, 
                        help="Number of teacher models (default: 3)")
    parser.add_argument("--teacher_epochs", type=int, default=5, 
                        help="Epochs to train each teacher (default: 5)")
    parser.add_argument("--student_epochs", type=int, default=20, 
                        help="Epochs to train student model (default: 20)")
    parser.add_argument("--temperature", type=float, default=3.0, 
                        help="Temperature for distillation (default: 3.0)")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = setup_config(args.dataset)
    logger.info(f"Testing distillation on {args.dataset} with {args.num_teachers} teachers")
    
    # Load data
    _, test_loader, unlabeled_loader, labeled_subset_loader = get_dataloader(
        dataset_name=config.dataset_name,
        num_clients=config.num_clients,
        unlabeled_ratio=config.unlabeled_ratio,
        alpha=0.5,  # Not important for this test
        batch_size=config.batch_size,
        labeled_subset_size=config.labeled_subset_size
    )
    
    # Create unlabeled data tensor
    unlabeled_data = torch.cat([batch for batch in unlabeled_loader], dim=0)
    
    # Create and train teacher models
    teachers = create_teacher_models(args.dataset, args.num_teachers)
    teachers = train_teacher_models(teachers, labeled_subset_loader, num_epochs=args.teacher_epochs)
    
    # Evaluate each teacher model
    logger.info("Evaluating teacher models")
    for i, teacher in enumerate(teachers):
        metrics = evaluate_model(teacher, test_loader)
        logger.info(f"Teacher {i+1} - Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
    
    # Generate soft targets from teachers
    logger.info("Generating soft targets from teachers")
    soft_targets = get_teacher_predictions(teachers, unlabeled_data, temperature=args.temperature)
    
    # Create training dataset with teacher logits
    distill_dataset = TensorDataset(unlabeled_data, soft_targets)
    
    # Get some labeled data for mixed training
    labeled_samples = []
    labeled_targets = []
    for batch in labeled_subset_loader:
        inputs, targets = batch
        labeled_samples.append(inputs)
        labeled_targets.append(targets)
    
    labeled_data = torch.cat(labeled_samples, dim=0)
    labeled_targets = torch.cat(labeled_targets, dim=0)
    
    # Important: Convert hard targets to one-hot encoding to match soft targets format
    num_classes = soft_targets.size(1)  # Get number of classes from soft targets
    one_hot_targets = F.one_hot(labeled_targets, num_classes=num_classes).float()
    
    # Create a labeled dataset for direct supervision with compatible format
    labeled_dataset = TensorDataset(labeled_data, one_hot_targets)
    
    # Combine distillation and direct supervision datasets
    combined_dataset = torch.utils.data.ConcatDataset([distill_dataset, labeled_dataset])
    
    # Create training dataloader
    train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Train student model
    logger.info("Training student model with combined distillation and direct supervision")
    student = create_student_model(args.dataset)
    student, train_losses, val_losses, val_accuracies = train_student_model(
        student, train_loader, test_loader, 
        temperature=args.temperature, 
        num_epochs=args.student_epochs
    )
    
    # Final evaluation
    logger.info("Evaluating student model")
    metrics = evaluate_model(student, test_loader)
    logger.info(f"Student - Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, val_accuracies)

if __name__ == "__main__":
    main()
