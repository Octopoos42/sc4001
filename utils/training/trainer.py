import torch.optim as optim
from datetime import datetime
import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import torch.nn as nn
import math

class ModelTrainer:
    def __init__(self, model, model_name, device, criterion, config):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.criterion = criterion
        self.config = config

        # Create directory for saving models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('models', f"{model_name}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['initial_lr'],
            weight_decay=config['weight_decay']
        )

        # Initialize learning rate scheduler
        self.scheduler = CustomLRScheduler(
            self.optimizer,
            config['initial_lr'],
            config['max_lr'],
            config['warmup_epochs'],
            config['num_epochs']
        )

        # Initialize early stopping
        self.early_stopper = EarlyStopper(
            patience=config['patience'],
            min_delta=config['min_delta'],
            monitor=config['monitor'],
            restore_best=config['restore_best'],
            save_path=os.path.join(self.save_dir, 'best_model.pth')
        )

        # Initialize tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in tqdm(train_loader, desc='Training'):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Handle different model outputs robustly (model may return logits or (logits, extra))
            result = self.model(batch_X)
            outputs = result[0] if isinstance(result, (tuple, list)) else result

            loss = self.criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Handle different model outputs robustly (model may return logits or (logits, extra))
                result = self.model(batch_X)
                outputs = result[0] if isinstance(result, (tuple, list)) else result

                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training process"""
        print(f"\nStarting training for {self.model_name}...")

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()  # Removed val_loss parameter
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Early stopping
            should_stop = self.early_stopper.step(
                val_loss= val_loss, val_acc= val_acc,
                model= self.model, optimizer= self.optimizer
            )

            if should_stop:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

            # Print epoch results
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')

        '''
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pth')
                print(f'New best model saved! (Val Loss: {val_loss:.4f})')
        '''
        # after training optionally restore best
        if self.early_stopper.restore_best:
            restored = self.early_stopper.restore_best_model(self.model, self.optimizer, map_location=self.device)
            if restored:
                print('Restored best model from early stopper')
        
        # Save final model
        self.save_model('final_model.pth')

    def save_model(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }, path)

    def plot_training_history(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title(f'{self.model_name} - Training History (Loss)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title(f'{self.model_name} - Training History (Accuracy)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.show()

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc='Testing'):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Handle different model outputs robustly (model may return logits or (logits, extra))
                result = self.model(batch_X)
                outputs = result[0] if isinstance(result, (tuple, list)) else result

                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())

        # Print classification report
        print(f"\nClassification Report ({self.model_name}):")
        print(classification_report(true_labels, predictions))

        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({self.model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.show()

class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, max_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.initial_lr + 0.5 * (self.max_lr - self.initial_lr) * (1 + np.cos(progress * np.pi))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
    
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        target_one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed_target = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        return torch.mean(torch.sum(-smoothed_target * torch.log_softmax(pred, dim=1), dim=1))

class EarlyStopper:
    """
    Simple patience-based early stopper.

    Parameters
    - patience: epochs to wait without improvement
    - min_delta: minimum change to qualify as improvement
    - monitor: 'val_loss' or 'val_acc'
    - mode: 'min' for loss, 'max' for accuracy
    - restore_best: if True, keep best state for later restore
    - save_path: optional path to save best checkpoint to disk
    """
    def __init__(self, patience=5, min_delta=0.0, monitor='val_loss',
                 mode=None, restore_best=True, save_path=None):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        if mode is None:
            self.mode = 'min' if monitor == 'val_loss' else 'max'
        else:
            self.mode = mode
        self.restore_best = bool(restore_best)
        self.save_path = save_path

        self.best_score = math.inf if self.mode == 'min' else -math.inf
        self.counter = 0
        self.best_model_state = None
        self.best_optimizer_state = None

    def _is_improved(self, current):
        if self.mode == 'min':
            return current < (self.best_score - self.min_delta)
        else:
            return current > (self.best_score + self.min_delta)

    def step(self, val_loss=None, val_acc=None, model=None, optimizer=None):
        """
        Call after validation. Returns True if training should stop.
        Provide either val_loss or val_acc depending on monitor.
        Optionally pass model and optimizer to save best states.
        """
        if self.monitor == 'val_loss':
            current = float(val_loss)
        else:
            current = float(val_acc)

        if self._is_improved(current):
            self.best_score = current
            self.counter = 0
            if self.restore_best and model is not None:
                # store best states in memory
                self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if optimizer is not None:
                    self.best_optimizer_state = optimizer.state_dict()
            if self.save_path is not None and model is not None:
                # save checkpoint to disk
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    self.monitor: current
                }
                torch.save(checkpoint, self.save_path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best_model(self, model, optimizer=None, map_location=None):
        """
        Restore best model and optimizer states if available.
        Returns True if restore succeeded.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if optimizer is not None and self.best_optimizer_state is not None:
                optimizer.load_state_dict(self.best_optimizer_state)
            return True
        if self.save_path is not None and torch.exists(self.save_path):
            ckpt = torch.load(self.save_path, map_location=map_location)
            model.load_state_dict(ckpt['model_state_dict'])
            if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            return True
        return False