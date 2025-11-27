import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import SimpleTracker
from utils import compute_centroid, draw_trajectory


def process_video(video_path, output_video, output_csv, model):
    tracker = SimpleTracker()
    cap = cv2.VideoCapture(video_path)

    # Safety: check if video opens
    if not cap.isOpened():
        print(" Cannot open video (skipping):", video_path)
        return

    # Auto detect FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    frame_index = 0
    detections = []

    while True:
        ret, frame = cap.read()

        # If video ended or first frame missing
        if not ret:
            if frame_index == 0:
                print("No frame read — video may be empty/corrupt:", video_path)
            break

        # Initialize writer on first valid frame
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        # YOLO inference
        results = model.predict(frame, device="cpu", verbose=False)[0]

        ball_centroid = None
        detected = False

        # Ball detection loop
        for box in results.boxes.xyxy.cpu().numpy():
            cx, cy = compute_centroid(box)
            ball_centroid = (cx, cy)
            detected = True

            detections.append([
                frame_index, *box, cx, cy, 1  # 1 = visible
            ])

            # Draw detected ball
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # If no detection, still log
        if not detected:
            detections.append([frame_index, 0, 0, 0, 0, 0, 0, 0])

        # Tracker update
        if ball_centroid:
            traj = tracker.update(ball_centroid)
            frame = draw_trajectory(frame, traj)
        else:
            tracker.update(None)

        # Write output frame
        if out is not None:
            out.write(frame)

        frame_index += 1

    # Cleanup
    cap.release()
    if out is not None:
        out.release()

    # Save CSV
    df = pd.DataFrame(
        detections,
        columns=["frame", "x1", "y1", "x2", "y2", "cx", "cy", "visibility"]
    )
    df.to_csv(output_csv, index=False)
    print(" Annotations saved:", output_csv)


if __name__ == "__main__":
    input_folder = "input_videos"       # contains mp4 + mov files
    results_folder = "results"
    annotations_folder = "annotations"

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    # Load YOLO model once
    model = YOLO("cricket_yolov8.pt")

    # Detect all .mp4 and .mov files
    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.mp4', '.mov'))
    ]

    # Sort videos alphabetically/numerically
    for video_name in sorted(video_files):
        input_path = os.path.join(input_folder, video_name)

        base = os.path.splitext(video_name)[0]  # remove .mp4/.mov extension

        output_video = os.path.join(results_folder, f"{base}_output.mp4")
        output_csv = os.path.join(annotations_folder, f"{base}_output.csv")

        print(f"\n Processing video: {video_name} ...")
        process_video(input_path, output_video, output_csv, model)

    print("\n All videos processed!")
