Cricket Ball Detection & Trajectory Pipeline (YOLOv8 + Tracking)

This repository implements an end-to-end pipeline for cricket ball detection, centroid extraction, and trajectory visualization from cricket match videos using YOLOv8 and a simple tracker.
The system processes multiple input videos and produces both annotated videos and per-frame CSV detection files.

 Setup Instructions

1. Clone the repository
   git clone https://github.com/yourusername/cricket-ball-detection.git
   cd cricket-ball-detection

2. Create & activate a virtual environment
   python -m venv venv
   source venv/bin/activate # Linux / MacOS
   venv\Scripts\activate # Windows

3. Install dependencies
   pip install -r requirements.txt

 Project Structure
project/
│── code/
│ ├── train.py
│ ├── inference.py
│ ├── tracker.py
│ ├── utils.py
│ └── batch_process.py
│
│── data/ # your dataset (ignored by Git)
│ ├── train/
│ ├── val/
│ ├── test/
│ └── data.yaml
│
│── annotations/ # output CSV detections
│── results/ # output processed videos
│── requirements.txt
│── README.md
│── .gitignore

Your dataset is automatically excluded from GitHub using .gitignore.

 How to Run the Pipeline
🔹 1. Train the YOLOv8 model (transfer learning)
python code/train.py

🔹 2. Run inference on a folder containing multiple videos
python code/batch_process.py

Each input video from the folder (e.g., 1.mp4, 2.mp4, …, 15.mp4) produces:

results/1_output.mp4
results/2_output.mp4
...
annotations/1_output.csv
annotations/2_output.csv

Dependencies

Example requirements.txt:

ultralytics
opencv-python
pandas
numpy

Install them using:

pip install -r requirements.txt

Notes

The pipeline runs entirely on CPU, optimized for VSCode.

YOLOv8 pretrained weights are downloaded automatically.

Outputs include:

1. Ball centroid per frame

2. Frame-wise bounding box CSV

3. Trajectory overlay on video
