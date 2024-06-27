import os
import shutil

# Configuration



SOURCE_DIR = ""      # Source directory containing subfolders
DEST_DIR = "frames"    # Output directory for renamed images
VALID_IMAGE_EXTENSIONS = {".jpg", ".png"}

# Check if source directory exists
if not os.path.exists(SOURCE_DIR):
    print(f"Error: Source directory '{SOURCE_DIR}' not found. Please specify the correct path.")
    exit(1)

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Function to process subfolders and copy images with new names
def process_subfolder(subfolder_path, dest_dir):
    subfolder_name = os.path.basename(subfolder_path)

    for file_name in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file_name)
        file_ext = os.path.splitext(file_name)[1].lower()

        # Check if the file is a valid image
        if os.path.isfile(file_path) and file_ext in VALID_IMAGE_EXTENSIONS:
            new_file_name = f"{subfolder_name}_{file_name}"
            new_file_path = os.path.join(dest_dir, new_file_name)

            # Copy the file with the new name
            shutil.copy2(file_path, new_file_path) # `copy2` preserves metadata
            print(f"Copied: {file_path} to {new_file_path}")

# Main execution
for entry in os.listdir(SOURCE_DIR):
    subfolder_path = os.path.join(SOURCE_DIR, entry)
    if os.path.isdir(subfolder_path):  # Process only if it's a directory
        process_subfolder(subfolder_path, DEST_DIR)

print("Finished copying and renaming images.")
