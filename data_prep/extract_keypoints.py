# data_prep/extract_keypoints.py
import cv2, os, argparse
import numpy as np
import keypoints as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

def extract_from_video(video_path, out_dir, label, seq_len=None):
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(static_image_mode=False, 
                              model_complexity=1,
                              refine_face_landmarks=False,
                              enable_segmentation=False) as holistic:
        idx = 0
        frames_saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            keypoints = landmarks_to_vector(results)
            # Some frames might be None if no detection — zero pad
            if keypoints is None:
                keypoints = np.zeros((543*3,), dtype=np.float32)  # adjust if different
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{os.path.basename(video_path)}_f{idx}.npy"), keypoints)
            idx += 1
            frames_saved += 1
            if seq_len and frames_saved >= seq_len:
                break
    cap.release()

def landmarks_to_vector(results):
    # pose 33 x 3, face 468 x 3, hand L 21 x 3, hand R 21 x 3
    if not results:
        return None
    parts = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            parts.extend([lm.x, lm.y, lm.z])
    else:
        parts.extend([0.0]*33*3)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            parts.extend([lm.x, lm.y, lm.z])
    else:
        parts.extend([0.0]*468*3)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            parts.extend([lm.x, lm.y, lm.z])
    else:
        parts.extend([0.0]*21*3)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            parts.extend([lm.x, lm.y, lm.z])
    else:
        parts.extend([0.0]*21*3)
    # result length = (33+468+21+21)*3 = 1563 actually (note: some pipelines report different counts)
    return np.array(parts, dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="video file or folder")
    parser.add_argument("--out", required=True, help="output folder for .npy files")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    path = args.input
    if os.path.isdir(path):
        vids = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".mp4",".avi"))]
    else:
        vids = [path]
    for v in vids:
        extract_from_video(v, args.out, args.label)
    print("done")
