import os
from glob import glob
import shutil
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Configuration (Constants for better maintainability)
LABELED_FRAMES_DIR = ""          # Source directory for original frames - Adjust if needed
AUGMENTED_FRAME_DIR = "augmented_frames"  # Output directory
ORIGINAL_ANNOTATIONS_FILE = "annotations.csv"
AUGMENTED_ANNOTATIONS_FILE = "augmented_annotations.csv"
NUM_AUGMENTATIONS_PER_IMAGE = 10

# Check if source directories exist; exit if not
if not os.path.exists(LABELED_FRAMES_DIR):
    print(f"Error: Labeled frames directory '{LABELED_FRAMES_DIR}' not found. Please specify the correct path.")
    exit(1)
if not os.path.exists(ORIGINAL_ANNOTATIONS_FILE):
    print(f"Error: Annotations file '{ORIGINAL_ANNOTATIONS_FILE}' not found. Please specify the correct path.")
    exit(1)

# Ensure output directory exists (create if needed)
os.makedirs(AUGMENTED_FRAME_DIR, exist_ok=True)

# Load data (only frames that haven't been augmented yet)
existing_augmented_files = set(os.listdir(AUGMENTED_FRAME_DIR))
original_frames = [path for path in glob(os.path.join(LABELED_FRAMES_DIR, "*.png"))
                   if os.path.basename(path) not in existing_augmented_files]

annotations = pd.read_csv(ORIGINAL_ANNOTATIONS_FILE)


# --- Transformation Pipeline ---
class ResizeWithPad(transforms.Resize):
    def __call__(self, img):
        return F.pad(super().__call__(img), self._get_padding(img))
        # ... (rest of ResizeWithPad class as before)


transform = transforms.Compose([
    ResizeWithPad(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
])


# --- Augmentation and Annotation Logic ---
def augment_and_save(frame_path, annotations):
    frame_name = os.path.basename(frame_path)
    image = Image.open(frame_path).convert('RGB')

    # Find annotations for this frame
    frame_annotations = annotations[annotations['frame'] == frame_name]
    if frame_annotations.empty:
        return []  # Skip if no annotations found

    new_annotations = []
    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        augmented_image = transform(image)
        aug_name = f"{os.path.splitext(frame_name)[0]}_aug_{i}.png"
        transforms.ToPILImage()(augmented_image).save(os.path.join(AUGMENTED_FRAME_DIR, aug_name))

        # Update annotations
        new_annotation = frame_annotations.copy()
        new_annotation['frame'] = aug_name
        new_annotations.append(new_annotation)

    return new_annotations


# --- Main Execution ---
def main():
    all_new_annotations = []

    with ThreadPoolExecutor() as executor:  # Default max_workers is usually good
        results = executor.map(augment_and_save, original_frames, [annotations] * len(original_frames))

        for result in results:
            all_new_annotations.extend(result)
            gc.collect()  # Be explicit about cleanup

    # Save augmented annotations
    if all_new_annotations:
        pd.concat(all_new_annotations).to_csv(AUGMENTED_ANNOTATIONS_FILE, index=False)


if __name__ == "__main__":
    main()