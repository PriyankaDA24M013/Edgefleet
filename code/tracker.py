class SimpleTracker:
    def __init__(self):
        self.trajectory = []

    def update(self, centroid):
        self.trajectory.append(centroid)
        return self.trajectory
