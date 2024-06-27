import os
import time
from common.constants import VERBOSE, VIDEO_PATH, OUTPUT_VIDEO_PATH, VISUAL_RESULTS_DIR, RESULT_FILE
from cv.initialization import initialize_directories, initialize_inferencer, initialize_model
from cv.inference import run_inference
from cv.plotting import plot_trajectories, plot_velocity_acceleration
from cv.overlay import create_overlay_video


def main():
    start_time = time.time()

    initialize_directories()

    if not os.path.exists(VIDEO_PATH):
        raise Exception(f"Error: Video file not found at {VIDEO_PATH}")
    inferencer = initialize_inferencer()
    model, device = initialize_model()

    hoof_trajectories, hoof_statesList, fps, frame_width, frame_height = run_inference(
        inferencer, model, device, RESULT_FILE, VIDEO_PATH, VISUAL_RESULTS_DIR, OUTPUT_VIDEO_PATH, VERBOSE)

    # Plot results
    trajectories_smoothed = plot_trajectories(hoof_trajectories)
    plot_velocity_acceleration(trajectories_smoothed)

    # Create overlay video
    create_overlay_video(VIDEO_PATH, inferencer, RESULT_FILE, hoof_trajectories, hoof_statesList, fps, frame_width,
                         frame_height, VERBOSE,visual_results_dir=VISUAL_RESULTS_DIR)

    end_time = time.time() - start_time
    print(f"Total script time: {end_time:.2f} seconds")


if __name__ == '__main__':
    main()
