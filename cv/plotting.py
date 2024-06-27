import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from common.utils import plot_data
from common.constants import PLOT_DIR, RUN_TIMESTAMP
import os

def plot_trajectories(hoof_trajectories):
    keys = ['left_back', 'right_back', 'left_front', 'right_front']
    trajectories = {key: np.array([frame[key] for frame in hoof_trajectories]) for key in keys}
    trajectories_smoothed = {key: uniform_filter1d(trajectories[key], size=10, axis=0) for key in keys}

    plt.figure(figsize=(12, 8))
    plt.plot(trajectories_smoothed['right_back'][:, 0], trajectories_smoothed['right_back'][:, 1],
             label='Right Back Hoof Smoothed', color='blue')
    plt.plot(trajectories_smoothed['right_front'][:, 0], trajectories_smoothed['right_front'][:, 1],
             label='Right Front Hoof Smoothed', color='yellow')

    plt.title('Hoof Trajectories (Raw and Smoothed)')
    plt.xlabel('X Coordinate (Normalized)')
    plt.ylabel('Y Coordinate (Normalized)')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(PLOT_DIR, f"hoof_trajectory_raw_smoothed_{RUN_TIMESTAMP}.png"))
    plt.show()

    return trajectories_smoothed

def plot_velocity_acceleration(trajectories_smoothed):
    keys = ['left_back', 'right_back', 'left_front', 'right_front']
    velocity = {key: np.gradient(trajectories_smoothed[key], axis=0) for key in keys}
    velocity_smoothed = {key: uniform_filter1d(velocity[key], size=10, axis=0) for key in keys}
    acceleration = {key: np.gradient(velocity[key], axis=0) for key in keys}
    acceleration_smoothed = {key: uniform_filter1d(acceleration[key], size=10, axis=0) for key in keys}

    plot_data([
        np.linalg.norm(velocity_smoothed['right_back'], axis=1),
        np.linalg.norm(velocity_smoothed['right_front'], axis=1)],
        ['Right Back Hoof Velocity', 'Right Front Hoof Velocity'],
        'Hoof Velocity Over Time', 'Frame Number', 'Velocity (Normalized)',
        f"hoof_velocity_{RUN_TIMESTAMP}.png"
    )

    plot_data([
        np.linalg.norm(acceleration['right_back'], axis=1),
        np.linalg.norm(acceleration['right_front'], axis=1)],
        ['Right Back Hoof Acceleration', 'Right Front Hoof Acceleration'],
        'Hoof Acceleration Over Time', 'Frame Number', 'Acceleration (Normalized)',
        f"hoof_acceleration_{RUN_TIMESTAMP}.png"
    )
