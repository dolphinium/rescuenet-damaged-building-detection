import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define class labels
class_labels = {
    1: 'building-no-damage',
    2: 'damaged-building'  # Merging medium, major damage, and total destruction into one class
}

# Define colors for each class for visualization
colors = {
    1: (255, 0, 0),     # Blue for building-no-damage
    2: (255, 0, 255),   # Magenta for damaged-building
}

# Create directories
def create_directories(base_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)

# Convert mask to bounding boxes
def mask_to_bboxes(mask):
    bboxes = []
    for class_id in np.unique(mask):
        if class_id == 0:  # Skip background
            continue
        # Find contours for the current class
        contours, _ = cv2.findContours((mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if class_id in [4, 5, 6]:  # Convert damaged classes to a single class id 2
                class_id = 2
            elif class_id == 3:  # Building no damage class
                class_id = 1
            else:
                continue
            bboxes.append((class_id, x, y, x + w, y + h))
    return bboxes


def process_dataset(image_dir, mask_dir, labels_dir):
    """Process dataset and save bounding boxes to YOLO format."""
    for mask_filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            continue

        bboxes = mask_to_bboxes(mask)
        print(bbox)
        image_filename = mask_filename.replace('_lab.png', '.jpg')
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        h, w, _ = image.shape
        label_filename = image_filename.replace('.jpg', '.txt')
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                class_id, x_min, y_min, x_max, y_max = bbox
                # Convert to YOLO format (normalized coordinates)
                x_center = (x_min + x_max) / 2.0 / w
                y_center = (y_min + y_max) / 2.0 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                f.write(f"{class_id - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Move image to the corresponding directory
        #shutil.copy(image_path, image_dir)

# Set your paths
# base_dir = 'data'
replica_base = "../data/additional_subset"
create_directories(replica_base)

# Paths for images and labels
train_image_dir = os.path.join(replica_base, 'train/train-org-img')
train_mask_dir = os.path.join(replica_base, 'train/train-label-img')
train_labels_dir = os.path.join(replica_base, 'labels/train')

val_image_dir = os.path.join(replica_base, 'val/val-org-img')
val_mask_dir = os.path.join(replica_base, 'val/val-label-img')
val_labels_dir = os.path.join(replica_base, 'labels/val')

test_image_dir = os.path.join(replica_base, 'test/test-org-img')
test_mask_dir = os.path.join(replica_base, 'test/test-label-img')
test_labels_dir = os.path.join(replica_base, 'labels/test')

# Process datasets
process_dataset(train_image_dir, train_mask_dir, train_labels_dir)
process_dataset(val_image_dir, val_mask_dir, val_labels_dir)
process_dataset(test_image_dir, test_mask_dir, test_labels_dir)