import cv2
import numpy as np
from pathlib import Path
from brand_detector import BrandDetector  # From our previous implementation

def test_single_image(image_path: str, model_path: str, output_path: str = None):
    """
    Test brand detection on a single image
    
    Args:
        image_path: Path to test image
        model_path: Path to trained model weights
        output_path: Path to save annotated image (optional)
    """
    # Initialize detector
    detector = BrandDetector(model_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Process image
    annotated_image, detections = detector.process_image(image)
    
    # Draw additional information
    for detection in detections:
        # Get detection info
        box = detection['bbox']
        conf = detection['confidence']
        label = f"{detection['class_name']} {conf:.2f}"
        
        # Draw label below the box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = box[0]
        text_y = box[3] + text_size[1] + 5  # Position below the box
        
        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, annotated_image)
    else:
        cv2.imshow('Detection Result', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections

if __name__ == "__main__":
    # Example usage
    test_single_image(
        image_path="path/to/test/image.jpg",
        model_path="path/to/trained/model.pt",
        output_path="detection_result.jpg"
    )