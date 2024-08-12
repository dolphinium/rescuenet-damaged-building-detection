import os

# Define the path to the annotations file
annotations_file = "bboxes/yolo_annotations_full.txt"

# Create directories if they do not exist
os.makedirs('labels/train', exist_ok=True)
os.makedirs('labels/val', exist_ok=True)
os.makedirs('labels/test', exist_ok=True)

# Function to get the new class label
def get_new_class_label(old_class):
    if old_class == 3:
        return 0
    elif old_class in [4, 5, 6]:
        return 1
    else:
        return None

# Dictionary to keep track of the images processed
processed_images = {}

# Read the annotations file
with open(annotations_file, 'r') as file:
    lines = file.readlines()

# Process each line
for line in lines:
    parts = line.strip().split()
    image_path = parts[0]
    class_label = int(parts[1])
    bbox_info = parts[2:]

    # Get the new class label
    new_class_label = get_new_class_label(class_label)
    if new_class_label is None:
        # Track images that should have empty annotation files
        if image_path not in processed_images:
            processed_images[image_path] = []
        continue

    # Extract the directory and file name
    directory = image_path.split('/')[-3]  # Adjusted to get the correct subdirectory
    file_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'  # Fixed file name extraction
    output_dir = f'labels/{directory}'
    output_path = os.path.join(output_dir, file_name)

    # Track the bounding box information for the image
    if image_path not in processed_images:
        processed_images[image_path] = []
    processed_images[image_path].append(f"{new_class_label} {' '.join(bbox_info)}\n")

# Write the bounding box information to the corresponding files
for image_path, bboxes in processed_images.items():
    directory = image_path.split('/')[-3]
    file_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    output_dir = f'labels/{directory}'
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w') as out_file:
        for bbox in bboxes:
            out_file.write(bbox)

# Ensure every image has a corresponding text file even if empty
all_images = set()
for line in lines:
    image_path = line.strip().split()[0]  # Corrected index to get the image path
    all_images.add(image_path)

for image_path in all_images:
    directory = image_path.split('/')[-3]
    file_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    output_dir = f'labels/{directory}'
    output_path = os.path.join(output_dir, file_name)
    
    # Create an empty file if it does not exist
    if not os.path.exists(output_path):
        open(output_path, 'w').close()

print("Processing complete.")