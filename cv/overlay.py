import cv2
import numpy as np
from common.constants import FADE_FRAMES, OVERLAY_VIDEO_PATH
from cv.video_utils import initialize_video_capture, blend_color
from cv.processing import determine_gait

def create_overlay_video(VIDEO_PATH, inferencer, result, hoof_trajectories, hoof_statesList, fps, frame_width, frame_height, VERBOSE,
                         visual_results_dir=None):
    cap, _ = initialize_video_capture(VIDEO_PATH, result, inferencer,visual_results_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    overlay_out = cv2.VideoWriter(OVERLAY_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    colors = {'left_back': (0, 255, 0), 'right_back': (0, 0, 255), 'left_front': (255, 0, 0), 'right_front': (255, 255, 0)}
    keys = ['left_back', 'right_back', 'left_front', 'right_front']

    for frame_num in range(len(hoof_trajectories)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        for i in range(1, frame_num + 1):
            neck_prev = hoof_trajectories[i - 1]['neck']
            neck = hoof_trajectories[i]['neck']

            for key in keys:
                prev_hoof = (int(hoof_trajectories[i - 1][key][0] + neck_prev[0]), int(hoof_trajectories[i - 1][key][1] + neck_prev[1]))
                current_hoof = (int(hoof_trajectories[i][key][0] + neck[0]), int(hoof_trajectories[i][key][1] + neck[1]))

                alpha = max(0, 1 - (frame_num - i) / FADE_FRAMES)
                if alpha > 0:
                    color = blend_color(colors[key], alpha)
                    cv2.line(frame, prev_hoof, current_hoof, color, 2)

        neck = hoof_trajectories[frame_num]['neck']
        for key in keys:
            hoof = (int(hoof_trajectories[frame_num][key][0] + neck[0]), int(hoof_trajectories[frame_num][key][1] + neck[1]))
            cv2.circle(frame, hoof, 5, colors[key], -1)

        hoof_trajectories_slice = hoof_trajectories[max(0, frame_num - FADE_FRAMES):frame_num + 1]
        gait = determine_gait(hoof_trajectories_slice)
        cv2.putText(frame, gait, (frame_width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        state_hoove = hoof_statesList[frame_num]
        for i, (key, state) in enumerate(state_hoove.items()):
            cv2.putText(frame, f"{key}: {state}", (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        overlay_out.write(frame)

    cap.release()
    overlay_out.release()
    cv2.destroyAllWindows()
