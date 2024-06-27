import os
import cv2
import pickle


def initialize_video_capture(video_path, result_path, inferencer, visual_results_dir):
    """Attempt to load results and video capture, infer if necessary."""
    try:
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
    except FileNotFoundError:
        result_generator = inferencer(video_path, vis_out_dir=visual_results_dir)
        result = list(result_generator)
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

    cap = cv2.VideoCapture(os.path.join(visual_results_dir, os.path.basename(video_path)))
    return cap, result


def blend_color(color, alpha):
    """Blend the given color with white based on the alpha value."""
    return tuple(int(c * alpha + 255 * (1 - alpha)) for c in color)
