# src/data_preparation/convert_labels.py
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_coordinates_to_yolo(coords):
    """
    Convierte coordenadas de puntos a formato YOLO (centro_x, centro_y, ancho, alto)
    Args:
        coords: lista de coordenadas [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        tuple: (centro_x, centro_y, ancho, alto)
    """
    # Convertir la lista de strings a floats
    coords = [float(x) for x in coords]
    
    # Extraer coordenadas x e y
    x_coords = coords[::2]  # [x1, x2, x3, x4]
    y_coords = coords[1::2]  # [y1, y2, y3, y4]
    
    # Calcular centro y dimensiones
    x_center = sum(x_coords) / len(x_coords)
    y_center = sum(y_coords) / len(y_coords)
    
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    return x_center, y_center, width, height

def convert_label_file(file_path):
    """
    Convierte un archivo de etiquetas al formato YOLO
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            coordinates = parts[1:]
            
            if len(coordinates) != 8:
                logger.error(f"Formato incorrecto en {file_path}: {line}")
                continue
                
            x_center, y_center, width, height = convert_coordinates_to_yolo(coordinates)
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Guardar el archivo convertido
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
            
        logger.info(f"Convertido: {file_path}")
        
    except Exception as e:
        logger.error(f"Error procesando {file_path}: {str(e)}")

def convert_dataset_labels(data_path):
    """
    Convierte todas las etiquetas en el dataset
    """
    data_path = Path(data_path)
    
    for split in ['train', 'valid', 'test']:
        labels_dir = data_path / split / 'labels'
        if not labels_dir.exists():
            continue
            
        logger.info(f"\nProcesando etiquetas en {split}")
        label_files = list(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            convert_label_file(label_file)

if __name__ == "__main__":
    # Obtener la ruta del proyecto
    current_path = Path(__file__).parent.parent.parent
    data_path = current_path / 'data'
    # Convertir todas las etiquetas
    convert_dataset_labels(data_path)
    logger.info("Conversión completada")