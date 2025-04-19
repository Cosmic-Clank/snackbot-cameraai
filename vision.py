import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from config import CONFIDENCE_THRESHOLD, DISTANCE_MIN, DISTANCE_MAX, YOLO_MODEL_PATH, TARGET_CLASS_ID


class VisionSystem:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        self.model = YOLO(YOLO_MODEL_PATH)
        self.triggered = False

    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        original = color_image.copy()
        return color_image, original, depth_frame

    def detect_and_track_people(self, color_image):
        # Use YOLO's tracking mode with persistence
        results = self.model.track(color_image, persist=True, verbose=False,
                                   conf=CONFIDENCE_THRESHOLD, classes=[TARGET_CLASS_ID])[0]
        return results.boxes
