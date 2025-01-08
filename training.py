from ultralytics import YOLO
import os
from pathlib import Path
import shutil
import yaml
import logging
from sklearn.model_selection import train_test_split

class BrandTrainingSetup:
    def __init__(self, project_dir: str = "brand_detection"):
        """
        Initialize training setup
        
        Args:
            project_dir: Directory for project files
        """
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "dataset"
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Create directory structure
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_dataset_structure(self):
        """Create the necessary directory structure for training"""
        # Create train/val/test directories
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)
            
    def create_data_yaml(self, class_names: list):
        """
        Create the data.yaml file required for YOLO training
        
        Args:
            class_names: List of class names (brands) to detect
        """
        data_yaml = {
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'test': str(self.images_dir / 'test'),
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(self.project_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
            
        self.logger.info(f"Created data.yaml with classes: {class_names}")
        
    def split_dataset(self, train_size=0.7, val_size=0.2, test_size=0.1):
        """
        Split the dataset into train/val/test sets
        
        Args:
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
        """
        # Get all image files
        image_files = list(Path(self.images_dir).glob('*.jpg')) + \
                     list(Path(self.images_dir).glob('*.png'))
        
        # Split into train/val/test
        train_files, temp_files = train_test_split(
            image_files, train_size=train_size, random_state=42
        )
        
        relative_val_size = val_size / (val_size + test_size)
        val_files, test_files = train_test_split(
            temp_files, train_size=relative_val_size, random_state=42
        )
        
        # Move files to appropriate directories
        for img_path in train_files:
            shutil.move(str(img_path), str(self.images_dir / 'train' / img_path.name))
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / 'train' / label_path.name))
                
        for img_path in val_files:
            shutil.move(str(img_path), str(self.images_dir / 'val' / img_path.name))
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / 'val' / label_path.name))
                
        for img_path in test_files:
            shutil.move(str(img_path), str(self.images_dir / 'test' / img_path.name))
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / 'test' / label_path.name))
                
    def train_model(self, epochs=100, batch_size=16, imgsz=640):
        """
        Train the YOLO model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
        """
        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(
            data=str(self.project_dir / 'data.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=50,  # Early stopping patience
            save=True,  # Save checkpoint
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.logger.info(f"Training completed. Results saved in {model.export()}")
        return results