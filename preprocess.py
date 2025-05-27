from tqdm import tqdm
from PIL import Image
import os
import shutil
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

def flatten_directory(root_dir, output_dir):
    
    subfolders = os.listdir(root_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Loop through all subdirectories and files
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder, "groundtruth")
        filenames = os.listdir(subfolder_path)
        for filename in tqdm(filenames, desc=f"Processing {subfolder}"):
            # Build relative path and convert to underscore format
            src = os.path.join(subfolder_path, filename)
            dest = os.path.join(output_dir, filename)
            shutil.copy2(src, dest)
            

def resize_image(file_path, output_dir, size):
    filename = os.path.basename(file_path)
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            with Image.open(file_path) as img:
                if "GroundTruth" in filename:
                    resized_img = img.resize(size, Image.NEAREST)
                else:
                    resized_img = img.resize(size, Image.BILINEAR)
                resized_img.save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def resize_dataset(root_dir, output_dir, size=(512, 512), max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_image, file_path, output_dir, size)
                   for file_path in files]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass  # Progress bar updates as tasks complete

    print("Resizing complete!")
            

def sort_files(root_dir, output_dir):
    masks_dir = os.path.join(output_dir, "masks")
    images_dir = os.path.join(output_dir, "images")
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Process files in the source directory
    for filename in tqdm(os.listdir(root_dir)):
        file_path = os.path.join(root_dir, filename)

        # Skip directories, process only files
        if os.path.isfile(file_path):
            if filename.endswith("_GroundTruth_color.png"):
                shutil.move(file_path, os.path.join(masks_dir, filename))
            elif not (filename.endswith("_GroundTruth_iMap.png") or filename.endswith("_GroundTruth_color.png")):
                shutil.move(file_path, os.path.join(images_dir, filename))

    print("Files sorted successfully!")
    
    
def preprocess_groundtruth(input_dir, max_workers=8):
    # Create the output subdirectories
    for subdir in ["0", "1", "2"]:
        os.makedirs(os.path.join(input_dir, subdir), exist_ok=True)

    def process_image_variant(input_path, output_path, target_color):
        try:
            image = Image.open(input_path).convert("RGB")
            pixels = image.load()
            width, height = image.size

            for x in range(width):
                for y in range(height):
                    r, g, b = pixels[x, y]

                    if target_color == "black" and (r, g, b) == (0, 0, 0):
                        pixels[x, y] = (255, 255, 255)
                    elif target_color == "green" and (r, g, b) == (0, 255, 0):
                        pixels[x, y] = (255, 255, 255)
                    elif target_color == "red" and (r, g, b) == (255, 0, 0):
                        pixels[x, y] = (255, 255, 255)
                    else:
                        pixels[x, y] = (0, 0, 0)

            image.save(output_path)
        except Exception as e:
            print(f"Error processing {input_path} ({target_color}): {e}")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(input_dir, filename)

                tasks.append(executor.submit(process_image_variant, input_path, os.path.join(input_dir, "0", filename), "black"))
                tasks.append(executor.submit(process_image_variant, input_path, os.path.join(input_dir, "1", filename), "green"))
                tasks.append(executor.submit(process_image_variant, input_path, os.path.join(input_dir, "2", filename), "red"))

        # Use tqdm to show progress
        for _ in tqdm(tasks):
            _.result()  # Wait for task to complete

    print("Processing complete! Images are saved in the 'masks' directory.")
    
    
def delete_black_images(input_path):
    masks_path = os.path.join(input_path, "masks")
    images_path = os.path.join(input_path, "images")

    # Paths for subfolders
    mask_0_path = os.path.join(masks_path, "0")
    mask_1_path = os.path.join(masks_path, "1")
    mask_2_path = os.path.join(masks_path, "2")

    # Loop through images in "0"
    for filename in os.listdir(mask_0_path):
        if filename.endswith("_GroundTruth_color.png"):  # Ensure correct file format
            file_path = os.path.join(mask_0_path, filename)

            # Load image as grayscale
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Check if all pixels are white (255)
            if np.all(img == 255):
                print(f"Deleting white image: {filename}")
                os.remove(file_path)  # Delete the white image

                # Extract image_id (remove "_GroundTruth_color.png")
                image_id = filename.replace("_GroundTruth_color.png", ".png")

                # Paths to corresponding images in other folders
                to_delete = [
                    os.path.join(mask_1_path, filename),
                    os.path.join(mask_2_path, filename),
                    os.path.join(images_path, image_id)  # Image in "images" folder
                ]

                # Delete corresponding images if they exist
                for file in to_delete:
                    if os.path.exists(file):
                        print(f"Deleting related file: {file}")
                        os.remove(file)
                        
                        
def create_test_set(input_dir, test_dir):
    # Base directories

    # Subdirectories
    subdirs = ["images", "masks/0", "masks/1", "masks/2"]

    # Create target directories if they don't exist
    for subdir in subdirs:
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)

    # Function to move matching files
    def move_files(source_folder, target_folder, prefix_list):
        for filename in os.listdir(source_folder):
            if any(filename.startswith(prefix) for prefix in prefix_list):
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                print(f"Moving {src_path} -> {dst_path}")
                shutil.move(src_path, dst_path)

    # Prefixes to filter (image IDs starting with 001 or 003)
    prefixes = ["003"]

    # Move files for each category
    for subdir in subdirs:
        src_dir = os.path.join(input_dir, subdir)
        dst_dir = os.path.join(test_dir, subdir)
        move_files(src_dir, dst_dir, prefixes)
        
        
def create_val_set(input_dir, val_dir):

    # Subdirectories
    subdirs = ["images", "masks/0", "masks/1", "masks/2"]

    # Create target directories if they don't exist
    for subdir in subdirs:
        os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)
        
        
    all_files = os.listdir(f"{input_dir}/images")
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # Function to move matching files
    def move_files(source_folder, target_folder, val_files):
        for filename in os.listdir(source_folder):
            if any(val_file in filename for val_file in val_files):
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                print(f"Moving {src_path} -> {dst_path}")
                shutil.move(src_path, dst_path)

    # Move files for each category
    for subdir in subdirs:
        src_dir = os.path.join(input_dir, subdir)
        dst_dir = os.path.join(val_dir, subdir)
        move_files(src_dir, dst_dir, val_files)

                    
if __name__ == "__main__":
    # Example usage
    root_dir = "data/weedmap/raw/RedEdge/RedEdge"
    
    flat_dir = "data/weedmap/UKAN/RedEdge_flat"
    resized_dir = "data/weedmap/UKAN/RedEdge_resized"
    train_dir = "data/weedmap/UKAN/train"
    
    val_dir = "data/weedmap/UKAN/val"
    test_dir = "data/weedmap/UKAN/test"
    
    # print("Flattening directory...")
    # flatten_directory(root_dir, flat_dir)
    
    print("Resizing dataset...")
    resize_dataset(flat_dir, resized_dir, size=(512, 512))
    
    print("Sorting files...")
    sort_files(resized_dir, train_dir)
    
    input_mask_dir = os.path.join(train_dir, "masks")
    preprocess_groundtruth(input_mask_dir)
    print("Ground truth preprocessing complete!")
    
    delete_black_images(train_dir)
    print("Black images deleted!")
    
    create_test_set(train_dir, test_dir)
    print("Test set created!")
    
    print("All preprocessing steps completed!")
