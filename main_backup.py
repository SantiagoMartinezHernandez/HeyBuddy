import cv2
import mediapipe as mp
import json
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task") # Basic parameters.
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, #we want to analyze video
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.PoseLandmarker.create_from_options(options)


LM_NAMES = LANDMARK_NAMES = [ #Dictionary of the 33 avaiable landmarks of the body
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

POSE_CONNECTIONS = [ # the coordinates corresponding to the landmarks
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Shoulders & Arms
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),           # Legs
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32) # Feet
]
def process_frame(frame, timestamp_ms):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # We put  the frame in the proper RGB system
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    res = landmarker.detect_for_video(mp_img,timestamp_ms)
    pose_json = {}
    if res.pose_landmarks: #We have data!
        lm_list = []
        for i, lm in enumerate(res.pose_landmarks[0]):
            lm_list.append({
                "id":i,
                "name":LM_NAMES[i],
                "x": round(lm.x,4),
                "y":round(lm.y,4),
                "z":round(lm.z,4),
                "visibility":round(lm.visibility,4) if hasattr(lm,"visibility") else 1.0
            })
        pose_json = {
            "session_id": "local_test",
            "timestamp_ms": int(time.time() * 1000),
            "landmarks": lm_list
        }

        # A simple demo :)
        draw_custom_landmarks(frame,lm_list,draw_connections=True)
        try:
            hip, knee, ankle = lm_list[23],lm_list[25],lm_list[27]
            knee_angle = calc_3d_angle(hip,knee,ankle)
            print(f"Demo knee angle: {knee_angle:.2f}°")
            cv2.putText(frame, f"Knee: {int(knee_angle)} °",
                        (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        except IndexError:
            pass
        return frame, pose_json
def calc_3d_angle (a,b,c):
    """
    Calculates the angle of a point "b" with respect to the others a and c
    :return: the angle (rads) between the three points
    """
    a = np.array([a["x"],a["y"],a["z"]]) # We translate all coordinates to np
    b = np.array([b["x"],b["y"],b["z"]])
    c = np.array([c["x"],c["y"],c["z"]])
    #Since these are now points, we can actually build the vectors
    ba = a - b
    bc = c - b

    #Then we can apply the dot product to get the cosine of the angle
    cos_ang = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_ang,-1.0,1))
    return np.degrees(angle)

def draw_custom_landmarks(img,lm_list, draw_connections=False):
    """
    Draws the landmarks cause that's fucking cool
    """
    h, w, _ = img.shape
    interest_ids = {23,24,25,26,27,28}
    coords = {}
    for lm in lm_list:
        if lm['visibility'] > 0.5:
            coords[lm['id']] = (int(lm['x'] * w), int(lm['y'] * h))

    if draw_connections:
        for start_id, end_id in POSE_CONNECTIONS:
            if start_id in coords and end_id in coords:
                cv2.line(img, coords[start_id], coords[end_id], (255, 255, 255), 2)

    for lm_id, pt in coords.items():
        if lm_id in interest_ids:
            # High-visibility green for ART/Wear primary sensors
            cv2.circle(img, pt, 8, (0, 255, 0), -1)
            cv2.circle(img, pt, 8, (255, 255, 255), 2)
        else:
            # Subtle blue for auxiliary points
            cv2.circle(img, pt, 4, (255, 0, 0), -1)

cap = cv2.VideoCapture(0)
print("Capture starts")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    timestamp_ms = int(time.time()*1000)
    annots_frame, current_json = process_frame(frame,timestamp_ms)
    cv2.imshow("IRON Debugger",annots_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()