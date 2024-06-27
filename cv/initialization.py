import os
import torch
from mmpose.apis import MMPoseInferencer
from common.constants import OUTPUT_DIR, PLOT_DIR, BEST_MODEL_PATH, VIDEO_DIR, RESULT_PATH
from hoovenet.model import HoovesModel
from hoovenet.utils import load_weights

def initialize_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)

def initialize_inferencer():
    return MMPoseInferencer(pose2d='animal', show_progress=True,device="cpu")

def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HoovesModel().to(device)
    model = load_weights(model, BEST_MODEL_PATH)
    model.eval()
    return model, device
