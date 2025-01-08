import argparse
import logging
from pathlib import Path
import sys
from brand_detector import BrandDetector
from training import BrandTrainingSetup
from test import test_single_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandDetectionCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Brand Detection CLI')
        self.setup_arguments()
        
    def setup_arguments(self):
        subparsers = self.parser.add_subparsers(dest='command', help='Commands')
        
        # Setup dataset structure
        setup_parser = subparsers.add_parser('setup', help='Setup project structure')
        setup_parser.add_argument('--project-dir', type=str, default='brand_detection',
                                help='Project directory name')
        setup_parser.add_argument('--brands', nargs='+', required=True,
                                help='List of brand names to detect')
                                
        # Train model
        train_parser = subparsers.add_parser('train', help='Train the model')
        train_parser.add_argument('--project-dir', type=str, default='brand_detection',
                                help='Project directory name')
        train_parser.add_argument('--epochs', type=int, default=100,
                                help='Number of training epochs')
        train_parser.add_argument('--batch-size', type=int, default=16,
                                help='Training batch size')
                                
        # Test single image
        test_parser = subparsers.add_parser('test', help='Test on single image')
        test_parser.add_argument('--image', type=str, required=True,
                                help='Path to test image')
        test_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model weights')
        test_parser.add_argument('--output', type=str,
                                help='Path to save output image')
                                
    def run(self):
        args = self.parser.parse_args()
        
        if args.command == 'setup':
            logger.info(f"Setting up project in {args.project_dir}")
            setup = BrandTrainingSetup(args.project_dir)
            setup.create_dataset_structure()
            setup.create_data_yaml(args.brands)
            logger.info("Project structure created successfully")
            
        elif args.command == 'train':
            logger.info("Starting training")
            setup = BrandTrainingSetup(args.project_dir)
            setup.train_model(epochs=args.epochs, batch_size=args.batch_size)
            
        elif args.command == 'test':
            logger.info(f"Testing image: {args.image}")
            test_single_image(args.image, args.model, args.output)
            
        else:
            self.parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    cli = BrandDetectionCLI()
    cli.run()