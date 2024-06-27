import numpy as np


def get_keypoints(result, frame=0):
    """Get the keypoints for the frame."""
    predictions = result[frame]["predictions"][0]
    keypoints = predictions[0]["keypoints"]
    return keypoints

def extract_hoof_keypoints(keypoints):
    """Extract the hoof keypoints from the detected keypoints."""
    """Openmmpose Animal Inference Keypoints """
    hooves = {
        'left_back': keypoints[13],  # L_B_Paw
        'right_back': keypoints[16],  # R_B_Paw
        'left_front': keypoints[7],  # L_F_Paw
        'right_front': keypoints[10]  # R_F_Paw
    }
    neck = keypoints[3]  # Neck
    return hooves, neck



def determine_gait(hoof_trajectories, window_size=30, threshold=2):
    """
    Determine the gait of the horse based on the hoof trajectories over a window of frames.
    """
    if len(hoof_trajectories) < window_size:
        return "Unknown"

    recent_trajectories = hoof_trajectories[-window_size:]

    # Trying to determin what hooves move at the same time to get gait pattern
    keys = ['left_back', 'right_back', 'left_front', 'right_front']
    trajectories = {key: np.array([frame[key] for frame in recent_trajectories]) for key in keys}
    velocities = {key: np.diff(trajectories[key], axis=0) for key in keys}
    patterns = {key: np.sign(velocities[key]) for key in keys}

    # 1 eq moves forwards -1 moves backwards
    gait_patterns = {
        'Walk': [[1, 0], [0, 1], [-1, 0], [0, -1]],
        'Trot': [[1, -1], [-1, 1]],
        'Canter': [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1]],
        'Gallop': [[1, 0], [0, 1], [-1, 0], [0, -1]]
    }

    gait_scores = {gait: 0 for gait in gait_patterns}
    for key in keys:
        for gait, pattern_list in gait_patterns.items():
            for pattern in pattern_list:
                if any(np.all(pattern == patterns[key][i]) for i in range(len(patterns[key]))):
                    gait_scores[gait] += 1
    # Gait with max points gets selected
    determined_gait = max(gait_scores, key=gait_scores.get)
    return determined_gait if gait_scores[determined_gait] > threshold else "Unknown"
