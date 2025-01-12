# utils/image_downloader.py
import os
import requests
from pathlib import Path
import logging
from bing_image_downloader import downloader
import shutil

class ImageCollector:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.images_dir = self.project_dir / "data" / "images"
        self.raw_downloads_dir = self.project_dir / "temp_downloads"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_images(self, query: str, limit: int = 100):
        """
        Download images using Bing Image Downloader
        """
        self.logger.info(f"Downloading {limit} images for query: {query}")
        
        # Download images to temporary directory
        downloader.download(
            query,
            limit=limit,
            output_dir=str(self.raw_downloads_dir),
            adult_filter_off=False,
            force_replace=False,
            timeout=60
        )

        # Move and rename images to dataset directory
        downloaded_dir = self.raw_downloads_dir / query
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, image_path in enumerate(downloaded_dir.glob("*")):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                new_name = f"{query.replace(' ', '_')}_{idx:04d}{image_path.suffix}"
                shutil.copy2(
                    image_path,
                    self.images_dir / new_name
                )

        # Cleanup downloads directory
        shutil.rmtree(self.raw_downloads_dir)
        self.logger.info(f"Images downloaded and organized in {self.images_dir}")

if __name__ == "__main__":
    collector = ImageCollector("Computer_Vision_F5")
    # Download different variations of Coca-Cola logos
    collector.download_images("coca cola logo", limit=50)
    collector.download_images("coca cola can logo", limit=25)
    collector.download_images("coca cola bottle logo", limit=25)