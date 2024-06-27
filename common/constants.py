import os
from datetime import datetime

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
CV_DIR = os.path.join(BASE_DIR, 'cv')

# Data
LABELED_FRAMES_DIR = os.path.join(DATASET_DIR, 'labeled')
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations.csv')
FRAME_DIR = os.path.join(DATASET_DIR, 'raw')
LABELED_FRAMES_FILE = os.path.join(DATA_DIR, 'labeled_frames.txt')

#CV
OUTPUT_DIR = os.path.join(CV_DIR, 'output_videos')
PLOT_DIR = os.path.join(CV_DIR, 'plots')
MODEL_FOLDER = os.path.join(BASE_DIR, 'hoovenet', 'models')
BEST_MODEL_FOLDER = os.path.join(BASE_DIR, 'hoovenet', 'best_models')
VISUAL_RESULTS_DIR = os.path.join(CV_DIR, 'vis_results')
RESULT_PATH = os.path.join(CV_DIR, 'results')





# Video constants
VIDEO_NAME = 'horse-video.mp4'
VIDEO_DIR = 'videos'
VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_NAME)
RESULT_FILE = os.path.join(RESULT_PATH, f'result_{VIDEO_NAME.split(".")[0]}.pkl')
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f'output_{RUN_TIMESTAMP}.mp4')
OVERLAY_VIDEO_PATH = os.path.join(OUTPUT_DIR, f'overlay_{RUN_TIMESTAMP}.mp4')
FADE_FRAMES = 20  # Number of frames over which to fade the trajectories

# For output of processing pipeline
VERBOSE = True

# Model paths
BEST_MODEL_PATH = os.path.join(BEST_MODEL_FOLDER, f"hoofnet_best_result.pth")

# Training constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10000
PATIENCE = 20
