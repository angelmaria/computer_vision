# train.py
import argparse
import logging
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandTrainer:
    def __init__(self, project_dir: str = "."):  # Changed default to current directory
        self.project_dir = Path(project_dir).resolve()  # Get absolute path
        self.data_dir = self.project_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.models_dir = self.data_dir / "models"
        self.config_dir = self.project_dir / "configs"
        
        # Create all necessary directories
        for directory in [self.images_dir, self.labels_dir, 
                         self.models_dir, self.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
    def create_dataset_structure(self):
        """Create the necessary directory structure for training"""
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)
            
    def create_data_yaml(self, class_names: list):
        """Create the data.yaml file in the configs directory"""
        data_yaml = {
            'path': str(self.project_dir),
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'test': str(self.images_dir / 'test'),
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.config_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
    def split_dataset(self, train_size=0.7, val_size=0.2, test_size=0.1):
        """Split dataset into train/val/test sets"""
        image_files = list(self.images_dir.glob('*.jpg')) + \
                     list(self.images_dir.glob('*.png'))
        
        train_files, temp_files = train_test_split(
            image_files, train_size=train_size, random_state=42
        )
        
        relative_val_size = val_size / (val_size + test_size)
        val_files, test_files = train_test_split(
            temp_files, train_size=relative_val_size, random_state=42
        )
        
        # Move files to appropriate directories
        for files, split in [(train_files, 'train'), 
                           (val_files, 'val'), 
                           (test_files, 'test')]:
            for img_path in files:
                shutil.move(str(img_path), 
                          str(self.images_dir / split / img_path.name))
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.move(str(label_path), 
                              str(self.labels_dir / split / label_path.name))
                    
    def train_model(self, epochs=100, batch_size=16, imgsz=640):
        """Train the YOLO model with additional error checking"""
        yaml_path = self.config_dir / 'data.yaml'
        
        # Check if data.yaml exists, create if it doesn't
        if not yaml_path.exists():
            logger.warning("data.yaml not found. Creating with default configuration...")
            self.create_data_yaml(['coca_cola'])
        
        # Verify yaml file is valid
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                required_keys = ['path', 'train', 'val', 'test', 'nc', 'names']
                missing_keys = [key for key in required_keys if key not in yaml_content]
                if missing_keys:
                    raise ValueError(f"data.yaml is missing required keys: {missing_keys}")
        except Exception as e:
            logger.error(f"Error reading data.yaml: {e}")
            raise
        
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=50,
            save=True,
            project=str(self.models_dir),
            name='train'
        )
        
        # Copy best model to standard location
        best_model = self.models_dir / 'train' / 'weights' / 'best.pt'
        if best_model.exists():
            shutil.copy(best_model, self.models_dir / 'best.pt')
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Brand Detection Training')
    parser.add_argument('--project-dir', type=str, default='.',  # Changed default to current directory
                        help='Project directory path')
    parser.add_argument('--brands', nargs='+', default=['coca_cola'],
                        help='List of brand names to detect')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    
    args = parser.parse_args()
    
    trainer = BrandTrainer(args.project_dir)
    trainer.create_dataset_structure()
    trainer.create_data_yaml(args.brands)
    trainer.split_dataset()
    trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()