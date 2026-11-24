import os
import yaml
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Path to the dataset directory and configuration file
DATASET_DIR = "stars and constellation.v1i.yolo26"
YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")


@dataclass
class BoundingBox:
    """Stores parsed numerical YOLO annotation data in memory."""
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


class LazyImage:
    """Proxy object that holds the image path and loads the image only when requested."""
    def __init__(self, image_path: str):
        self._image_path = image_path

    def load(self) -> np.ndarray:
        """Reads and returns the image from disk as a NumPy array (BGR format)."""
        if not os.path.exists(self._image_path):
            raise FileNotFoundError(f"Image file not found: {self._image_path}")
        
        image = cv2.imread(self._image_path)
        if image is None:
            raise ValueError(f"Failed to decode image: {self._image_path}")
        
        return image


@dataclass
class DatasetItem:
    """Represents a single data sample containing a lazy-loaded image and eager-loaded labels."""
    image: LazyImage
    annotations: List[BoundingBox]


def _parse_yolo_label_file(label_path: str, class_names: List[str]) -> List[BoundingBox]:
    """Helper function to eagerly parse numerical coordinates from a YOLO .txt file."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
                
                annotations.append(
                    BoundingBox(
                        class_id=class_id,
                        class_name=class_name,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height
                    )
                )
    return annotations


def load_dataset_split(split: str) -> Tuple[List[DatasetItem], List[str]]:
    """
    Loads a dataset split ('train', 'valid', or 'test').
    Numerical annotations are eagerly parsed into memory, while images use lazy loading.
    """
    valid_splits = ("train", "valid", "test")
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"Configuration file not found at: {os.path.abspath(YAML_PATH)}")

    with open(YAML_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    class_names: List[str] = config.get("names", [])
    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {os.path.abspath(images_dir)}")

    dataset: List[DatasetItem] = []
    valid_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(images_dir, filename)
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            # Lazy load wrapper for the heavy image file
            lazy_img = LazyImage(image_path=image_path)
            
            # Eagerly parse numerical coordinates into memory
            annotations = _parse_yolo_label_file(label_path=label_path, class_names=class_names)

            dataset.append(DatasetItem(image=lazy_img, annotations=annotations))

    return dataset, class_names


# --- Testing block (prints only when script is run directly) ---
if __name__ == "__main__":
    train_dataset, classes = load_dataset_split("train")
    
    print(f"Loaded {len(train_dataset)} items from 'train' split.")
    print(f"Classes: {classes}\n")

    # Iterate through all loaded records and print their numerical annotations
    for idx, item in enumerate(train_dataset):
        print(f"Record [{idx}] ({len(item.annotations)} objects):")
        for box in item.annotations:
            print(f"  - {box.class_name} (ID: {box.class_id}): "
                  f"x={box.x_center:.4f}, y={box.y_center:.4f}, w={box.width:.4f}, h={box.height:.4f}")

    # Trigger lazy loading on a single sample image only to verify memory efficiency
    if train_dataset:
        sample_image_matrix = train_dataset[0].image.load()
        print(f"\nLazy-loaded sample image dimensions (Height, Width, Channels): {sample_image_matrix.shape}")
