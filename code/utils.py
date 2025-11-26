import cv2

def compute_centroid(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def draw_trajectory(frame, points):
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
    return frame
