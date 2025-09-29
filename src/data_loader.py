import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

def load_images(data_dir, image_size=(128, 128)):
    """
    Loads images from class subdirectories, resizes, and converts to grayscale.
    """
    print(f"Loading images from {data_dir}...")
    images = []
    labels = []
    
    # Iterate through class subdirectories (e.g., 'NORMAL', 'PNEUMONIA')
    for class_label, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            
            # Read the image
            image = imread(image_path)
            
            # Convert to grayscale if it has 3 channels (RGB)
            if image.ndim == 3:
                image = rgb2gray(image)
            
            # Resize the image to a fixed size
            image = resize(image, image_size, anti_aliasing=True)
            
            images.append(image)
            labels.append(class_label) # 0 for NORMAL, 1 for PNEUMONIA
            
    print(f"Loaded {len(images)} images successfully.")
    # Return NumPy arrays for easier processing later
    return np.array(images), np.array(labels)