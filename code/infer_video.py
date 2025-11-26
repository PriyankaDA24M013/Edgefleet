import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import SimpleTracker
from utils import compute_centroid, draw_trajectory

def process_video(video_path, output_video, output_csv, model):
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(video_path)

    # ✅ Debugging + safety: check if video opens
    if not cap.isOpened():
        print("❌ Cannot open video (skipping):", video_path)
        return

    # ✅ Auto detect FPS instead of hardcoding 30/30
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:  # fallback if invalid
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    frame_index = 0
    detections = []

    while True:
        ret, frame = cap.read()

        # ✅ Stop safely on first frame failure too
        if not ret:
            if frame_index == 0:
                print("⚠️ No frame read at start — video may be empty/corrupt (skipping):", video_path)
            break

        # ✅ Initialize writer only if we got at least one frame
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        # ✅ Stable inference using .predict()
        results = model.predict(frame, device="cpu", verbose=False)[0]

        ball_centroid = None
        detected = False

        # ✅ Ball detection loop
        for box in results.boxes.xyxy.cpu().numpy():
            cx, cy = compute_centroid(box)
            ball_centroid = (cx, cy)
            detected = True
            detected = True

            detections.append([frame_index, *box, cx, cy, 1])  # 1 = visible
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # mark ball center

        if not detected:
            # if no ball, still log frame with visibility 0
            detections.append([frame_index, 0, 0, 0, 0, 0, 0, 0])

        # ✅ Update tracker if ball visible
        if ball_centroid and detected:
            traj = tracker.update(ball_centroid)
            frame = draw_trajectory(frame, traj)
        else:
            tracker.update(None)  # optional: handle lost state inside tracker

        # ✅ Write frame if writer exists
        if out is not None:
            out.write(frame)

        frame_index += 1

    # ✅ Safe cleanup
    cap.release()
    if out is not None:
        out.release()

    # ✅ Save CSV with visibility flag
    df = pd.DataFrame(
        detections,
        columns=["frame", "x1", "y1", "x2", "y2", "cx", "cy", "visibility"]
    )
    df.to_csv(output_csv, index=False)
    print("📁 Annotations saved:", output_csv)


if __name__ == "__main__":
    input_folder = "input_videos"       # contains 1.mp4 to 15.mp4
    results_folder = "results"          # processed videos
    annotations_folder = "annotations"  # CSV files

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    # ✅ Load YOLO model once (your fine-tuned model)
    model = YOLO("cricket_yolov8.pt")

    # ✅ Process videos 1.mp4 → 15.mp4
    for i in range(1, 15 + 1):
        input_path = os.path.join(input_folder, f"{i}.mp4")
        output_video = os.path.join(results_folder, f"{i}_output.mp4")
        output_csv = os.path.join(annotations_folder, f"{i}_output.csv")

        print(f"\n🎬 Processing video {i}...")

        process_video(input_path, output_video, output_csv, model)

    print("\n✅ All videos processed!")
