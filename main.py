import cv2
import threading
import uvicorn
from api import app
from vision import VisionSystem
from llm import generate_suggestion
from config import CONFIDENCE_THRESHOLD, DISTANCE_MIN, DISTANCE_MAX, TARGET_CLASS_ID


def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Start API server in background
api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()

vision = VisionSystem()
current_person_id = None  # Track the current person's ID, persists until new ID

try:
    while True:
        color_image, original_frame, depth_frame = vision.get_frame()
        if color_image is None:
            continue

        tracked_objects = vision.detect_and_track_people(color_image)

        new_person_id = None  # ID of the closest person within distance range
        closest_distance = float('inf')
        closest_box = None
        closest_image = None

        # Log detected objects
        # print(f"\n[DEBUG] Detected {len(tracked_objects)} objects")
        for obj in tracked_objects:
            cls = int(obj.cls[0])
            conf = float(obj.conf[0])
            track_id = int(obj.id[0]) if obj.id is not None else None

            if cls != TARGET_CLASS_ID or conf < CONFIDENCE_THRESHOLD or track_id is None:
                continue

            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            distance = depth_frame.get_distance(cx, cy)

            # print(
            #     f"[DEBUG] ID: {track_id}, Conf: {conf:.2f}, Distance: {distance:.2f}m")

            # Check if person is within distance range and closest
            if DISTANCE_MIN <= distance <= DISTANCE_MAX and distance < closest_distance:
                closest_distance = distance
                new_person_id = track_id
                closest_box = obj
                closest_image = original_frame.copy()

            label = f"Person ID:{track_id} {conf:.2f} | {distance:.2f}m"
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Trigger LLM if person ID changes
        if new_person_id is not None and new_person_id != current_person_id and not vision.triggered:
            print(
                f"\n[EVENT] New person detected with ID {new_person_id} at {closest_distance:.2f}m")
            vision.triggered = True
            current_person_id = new_person_id

            def threaded_llm():
                generate_suggestion(closest_image)
                vision.triggered = False

            threading.Thread(target=threaded_llm).start()
        elif new_person_id is None:
            # print(
            #     f"[DEBUG] No person detected, keeping current_person_id as {current_person_id}")
            pass

        cv2.imshow("YOLO + RealSense", color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    vision.stop()
    cv2.destroyAllWindows()
