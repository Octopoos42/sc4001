import torch
from torch.utils.data import DataLoader
from utils.preprocessing import ECGDataset
from utils.training.trainer import ModelTrainer, SmoothCrossEntropyLoss, SmoothFocalLoss

def get_training_config():
    """
    Returns improved training configuration for both RNN and CNN models
    """
    config = {
        'CNN-RNN': {
            'model_params': {
                'input_channels': 1,
                'sequence_length': 187,
                'hidden_size': 128,
                'num_heads': 4,
                'num_classes': 5,
                'dropout': 0.3
            },
            'training_params': {
                'initial_lr': 5e-6,
                'max_lr': 1e-3,
                'batch_size': 128,
                'num_epochs': 100,
                'warmup_epochs': 30,
                'label_smoothing': 0.05,
                'weight_decay': 1e-3,
                'gradient_clip_val': 0.5,
                'patience': 20,
                'min_delta': 5e-4,
                'monitor': 'val_loss',
                'restore_best': True,
                'gamma': 2
            }            
        },
        'GRU': {
            'model_params': {
                'input_channels': 1,
                'hidden_size': 128,
                'num_layers': 2,
                'num_classes': 5
            },
            'training_params': {
                'initial_lr': 5e-6,
                'max_lr': 1e-3,
                'batch_size': 128,
                'num_epochs': 100,
                'warmup_epochs': 30,
                'label_smoothing': 0.05,
                'weight_decay': 1e-3,
                'gradient_clip_val': 0.5,
                'patience': 20,
                'min_delta': 5e-4,
                'monitor': 'val_loss',
                'restore_best': True,
                'gamma': 2
            }
        }
    }
    return config

def train_model(model, train_loader, val_loader, test_loader):
    """Train models"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get configurations
    config = get_training_config()
    model_name = model.model_name

    # Initialize criterion with label smoothing
    criterion = SmoothCrossEntropyLoss(smoothing=config[model_name]['training_params']['label_smoothing'])

    # Train model
    trainer = ModelTrainer(
        model=model,
        model_name=model_name,
        device=device,
        criterion=criterion,
        config=config[model_name]['training_params']
    )
    trainer.train(
        train_loader,
        val_loader,
        config[model_name]['training_params']['num_epochs']
    )
    trainer.plot_training_history()
    trainer.evaluate(test_loader)

    return trainer

def new_train_model(model, train_loader, val_loader, test_loader):
    """Train models"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get configurations
    config = get_training_config()
    model_name = model.model_name

    # Initialize criterion with label smoothing
    criterion = SmoothFocalLoss(smoothing=config[model_name]['training_params']['label_smoothing'], gamma=config[model_name]['training_params']['gamma'])

    # Train model
    trainer = ModelTrainer(
        model=model,
        model_name=model_name,
        device=device,
        criterion=criterion,
        config=config[model_name]['training_params']
    )
    trainer.train(
        train_loader,
        val_loader,
        config[model_name]['training_params']['num_epochs']
    )
    trainer.plot_training_history()
    trainer.evaluate(test_loader)

    return trainer