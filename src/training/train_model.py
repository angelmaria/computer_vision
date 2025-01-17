# src/training/train_model.py
from ultralytics import YOLO
import torch
import optuna
from pathlib import Path
import yaml
import logging
from datetime import datetime
import os
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_yaml():
    """Create the dataset.yaml with correct paths"""
    dataset_config = {
        'path': str(DATA_PATH),
        'train': str(DATA_PATH / 'train' / 'images'),
        'val': str(DATA_PATH / 'valid' / 'images'),
        'test': str(DATA_PATH / 'test' / 'images'),
        'names': {i: name for i, name in enumerate(BRAND_CLASSES)},
        'nc': len(BRAND_CLASSES)
    }
    
    with open(YAML_PATH, 'w') as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)
    
    return YAML_PATH

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameter search space
    params = {
        'lr0': trial.suggest_float('lr0', 1e-5, 1e-2, log=True),
        'lrf': trial.suggest_float('lrf', 1e-6, 1e-3, log=True),
        'momentum': trial.suggest_float('momentum', 0.7, 0.99),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
        'batch': trial.suggest_categorical('batch', [8, 16, 32]),
        'imgsz': trial.suggest_categorical('imgsz', [416, 512, 640]),
        'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW']),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5)
    }
    
    # Create unique run name for this trial
    run_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Initialize model
        model = YOLO(PRETRAINED_MODEL)
        
        # Training parameters
        train_args = {
            'data': str(create_dataset_yaml()),
            'epochs': 10,  # Reduced epochs for optimization
            'device': 'cpu' if not torch.cuda.is_available() else 'cuda',
            'project': str(RUNS_DIR),
            'name': run_name,
            'exist_ok': True,
            'patience': 5,  # Early stopping
            **params
        }
        
        # Train model
        results = model.train(**train_args)
        
        # Get the best validation metric
        best_map50 = max(results.results_dict.get('metrics/mAP50(B)', [0]))
        
        return best_map50
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

def train_model(optimize=False, n_trials=20, final_epochs=50):
    """
    Train the model with optional hyperparameter optimization
    Args:
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of optimization trials if optimize=True
        final_epochs: Number of epochs for final training
    """
    try:
        if optimize:
            logger.info("Starting hyperparameter optimization...")
            study = optuna.create_study(
                study_name="logo_detection",
                direction="maximize",
                storage="sqlite:///optuna_study.db",
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner()
            )
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            logger.info("\nBest hyperparameters:")
            logger.info(study.best_params)
            
            # Save best parameters
            best_params_path = RUNS_DIR / 'best_params.yaml'
            with open(best_params_path, 'w') as f:
                yaml.dump(study.best_params, f)
            
            train_args = {
                **study.best_params,
                'epochs': final_epochs,
            }
        else:
            train_args = {
                'epochs': final_epochs,
                'batch': 16,
                'imgsz': 640,
            }
        
        # Final training with best parameters or default
        logger.info("Starting final training...")
        model = YOLO(PRETRAINED_MODEL)
        
        final_args = {
            'data': str(create_dataset_yaml()),
            'device': 'cpu' if not torch.cuda.is_available() else 'cuda',
            'project': str(RUNS_DIR),
            'name': f"final_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            **train_args
        }
        
        results = model.train(**final_args)
        
        # Get path of best model
        best_model_path = Path(results.best)
        
        return best_model_path
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()
    
    model_path = train_model(optimize=args.optimize, n_trials=args.trials, final_epochs=args.epochs)
    logger.info(f"Training completed. Best model saved at: {model_path}")