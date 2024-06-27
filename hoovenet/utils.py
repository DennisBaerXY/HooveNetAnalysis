import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from common.constants import LABELED_FRAMES_DIR, ANNOTATIONS_FILE, BATCH_SIZE, BEST_MODEL_PATH
from hoovenet.model import HoovesModel


def model_predict(frame_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # model_predict is also being used with frame_path as a loaded image when used for a video
    if type(frame_path) == str:
        image = Image.open(frame_path).convert("RGB")
    else:

        image = frame_path
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    probabilities = torch.sigmoid(outputs).detach().cpu().numpy()[0]
    predictions = (probabilities > 0.5).astype(int)
    return {
        'left_back': predictions[0],
        'right_back': predictions[1],
        'left_front': predictions[2],
        'right_front': predictions[3]
    }

def get_dataloaders(batch_size=BATCH_SIZE):
    # Custom Dataset
    class HoofDataset(Dataset):
        def __init__(self, frame_dir, annotations_file, transform=None):
            self.frame_dir = frame_dir
            self.annotations = pd.read_csv(annotations_file)
            self.transform = transform

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            img_name = os.path.join(self.frame_dir, self.annotations.iloc[idx, 0])
            image = Image.open(img_name).convert("RGB")
            labels = self.annotations.iloc[idx, 1:].values.astype('float')

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(labels, dtype=torch.float32)

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure images are resized to the correct input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Apply normalization
    ])

    # Load Dataset
    hoof_dataset = HoofDataset(LABELED_FRAMES_DIR, ANNOTATIONS_FILE, transform=transform)

    # Split dataset into train, validation sets
    train_indices, val_indices = train_test_split(list(range(len(hoof_dataset))), test_size=0.3, random_state=27)

    train_dataset = Subset(hoof_dataset, train_indices)
    val_dataset = Subset(hoof_dataset, val_indices)

    # Create Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def load_weights(model,weight_path=BEST_MODEL_PATH):
    try:
        model.load_state_dict(torch.load(weight_path))
    except Exception as e:
        print(f"Couldnt load model with given weight_path: {weight_path}\nError message: {e}")
    return model

