import os
import numpy as np
import pandas as pd
from PIL import Image

def load_and_preprocess_images(image_dir, max_images=500, output_dir=None):
    """
    Load images from a directory and preprocess into [3, 224, 224] NumPy arrays.
    If output_dir is provided, save the processed tensors there.

    :param image_dir: str, image folder path
    :param max_images: int, max number of images to load
    :param output_dir: str, optional output directory for processed tensors
    :return: list, processed image arrays
    """
    processed_images = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Recursively find all image files
        image_files = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, f))
    except FileNotFoundError:
        print(f"Error: Directory not found at {image_dir}")
        return []

    if not image_files:
        print(f"Warning: No images found in {image_dir}")
        return []

    # Shuffle to get a good sample if there are many images
    np.random.shuffle(image_files)

    for i, image_path in enumerate(image_files):
        if i >= max_images:
            break
            
        try:
            with Image.open(image_path) as img:
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                
                # Handle RGBA images by slicing off the alpha channel
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                if img_array.shape[2] != 3:
                    print(f"Skipping {os.path.basename(image_path)}: not in RGB format (shape: {img_array.shape}).")
                    continue

                img_array = img_array.transpose((2, 0, 1))

                if output_dir:
                    base_filename = os.path.basename(image_path)
                    output_filename = os.path.join(output_dir, f"{base_filename}.npy")
                    np.save(output_filename, img_array)

                processed_images.append(img_array)

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
            
    return processed_images

def load_images_to_batches(image_dir, batch_size, max_batches=10):
    """
    Load images from a directory and pack into [batch_size, 3, 224, 224] batches.
    Only complete batches are returned.

    :param image_dir: str, image folder path
    :param batch_size: int, batch size
    :param max_batches: int, max number of batches (default 10)
    :return: list, complete batch arrays
    """
    # List of batches
    batches = []

    # Get image files in the directory
    image_files = [file for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Temporary list for the current batch
    current_batch = []

    if not image_files:
        return batches

    max_batches = int(max_batches) if max_batches is not None else 10
    if max_batches <= 0:
        return batches

    # Reuse images if needed until max_batches are filled
    i = 0
    while len(batches) < max_batches:
        image_file = image_files[i % len(image_files)]
        i += 1
        # Build full image path
        image_path = os.path.join(image_dir, image_file)
        try:
            # Open image
            with Image.open(image_path) as img:
                # Resize to (224, 224)
                img = img.resize((224, 224))
                
                # Convert to NumPy array
                img_array = np.array(img, dtype=np.float32)
                
                # Convert grayscale / single-channel to RGB
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)

                # Ensure RGB
                if img_array.shape[2] != 3:
                    raise ValueError(f"Image {image_file} is not in RGB format.")

                # Convert to [3, 224, 224]
                img_array = img_array.transpose((2, 0, 1))
                
                # Add to current batch
                current_batch.append(img_array)

                # Save batch when size is reached
                if len(current_batch) == batch_size:
                    batches.append(np.stack(current_batch))
                    current_batch = []

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Drop incomplete final batch

    # Keep only 10 batches; each scheduling window uses 10 batches
    return batches
def load_sst2_data(file_path):
    """Load text from a TSV file."""
    df = pd.read_csv(file_path, sep='\t')
    return df['sentence'].tolist()  # Return plain text list
def get_dataLoader(file_path,batchsize):
    batchsize=int(batchsize)
    # texts = read_imdb("./data/aclImdb/imdb_test.txt" )
    texts = load_sst2_data(file_path)
    # Ensure text count is a multiple of batch_size (drop remainder)
    truncated_length = len(texts) // batchsize * batchsize
    texts = texts[:truncated_length]
    text_batches = [texts[i:i+batchsize] for i in range(0, len(texts), batchsize)]
    return text_batches[:10]
