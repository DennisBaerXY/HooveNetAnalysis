import cv2
from PIL import Image
from hoovenet.utils import model_predict
from cv.video_utils import initialize_video_capture, blend_color
from cv.processing import get_keypoints, extract_hoof_keypoints, determine_gait


def run_inference(inferencer, model, device,result_path, VIDEO_PATH, VISUAL_RESULTS_DIR, OUTPUT_VIDEO_PATH, VERBOSE):

    cap, result = initialize_video_capture(VIDEO_PATH,result_path, inferencer, VISUAL_RESULTS_DIR)


    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    hoof_trajectories = []
    hoof_statesList = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = get_keypoints(result, frame=frame_num)
        hooves, neck = extract_hoof_keypoints(keypoints)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hoof_states = model_predict(image, model, device)
        hoof_statesList.append(hoof_states)

        norm_hooves = {key: (hoof[0] - neck[0], hoof[1] - neck[1]) for key, hoof in hooves.items()}
        norm_hooves['neck'] = neck
        hoof_trajectories.append(norm_hooves)

        gait = determine_gait(hoof_trajectories)
        colors = {'left_back': (0, 255, 0), 'right_back': (0, 0, 255), 'left_front': (255, 0, 0),
                  'right_front': (255, 255, 0), 'neck': (255, 0, 255)}
        if VERBOSE:
            for key, (x, y) in hooves.items():
                cv2.circle(frame, (int(x), int(y)), 5, colors[key], -1)
            cv2.putText(frame, gait, (frame_width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            for i, (key, state) in enumerate(hoof_states.items()):
                cv2.putText(frame, f"{key}: {state}", (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2)

            for key, state in hoof_states.items():
                if state == 1:
                    cv2.circle(frame, (int(hooves[key][0]), int(hooves[key][1])), 8, (0, 255, 0), -1)  # Green circle
                else:
                    cv2.circle(frame, (int(hooves[key][0]), int(hooves[key][1])), 8, (0, 0, 255), -1)  # Red circle

        out.write(frame)

        if VERBOSE:
            cv2.imshow('Output Video', frame)

        frame_num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return hoof_trajectories, hoof_statesList, fps, frame_width, frame_height
