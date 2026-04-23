import os
import cv2
import mediapipe as mp
import numpy as np

DISTANCE_THRESHOLD = 0.15  # threshold for identity focus
FRAME_WINDOW = 5  # number of frames to consider for smoothing
ANGLE_SQUAT_MAX = 100
ANGLE_BACK_MIN = 80
SIDEBAR_WIDTH = 350

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

try: 
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing


pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Capture video from the file
video_folder = "videos_squats/"
#List of video files to process
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

for video_name in video_files:
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    prev_center = None
    # Initialization of counters for smoothing
    counters = {"depth": 0, "back": 0, "valgus": 0}

    print (f"Processing video: {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        # --- CREATE SIDEBAR ---
        layout = np.zeros((h, w + SIDEBAR_WIDTH, 3), dtype=np.uint8)
        layout[:, :w] = frame


        # mediapipe requires RGB images, so convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and find pose landmarks
        results = pose.process(rgb_frame)

        warning_texts = []
        is_jump = False

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Filter step: calculate the current center (hips)
            curr_hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            curr_hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            curr_center = np.array([(curr_hip_l.x + curr_hip_r.x) / 2, (curr_hip_l.y + curr_hip_r.y) / 2])

            
            if prev_center is not None and np.linalg.norm(curr_center - prev_center) > DISTANCE_THRESHOLD:
                    # If the distance exceeds the threshold, we can consider this as a new person and ignore it
                    # we print a warning on the image
                    is_jump = True
            else: 
                    prev_center = curr_center
                    #cv2.putText(frame, "WARNING: detection jump: (other person?)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    


            # --- 2. DATA EXTRACTION ---
            # left side for profile
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y

            # right sife for face sight (valgus)
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y

            # --- 3. POSTURE ANALYSIS ---
            angle_knee = calculate_angle(l_hip, l_knee, l_ankle)
            angle_back = calculate_angle(l_shoulder, l_hip, l_knee)
            dist_knees = abs(l_knee[0] - r_knee[0])
            dist_hips = abs(l_hip[0] - r_hip[0])

            # --- 4. COUNTERS ---
            # depth counter
            counters["depth"] = counters["depth"] + 1 if angle_knee > ANGLE_SQUAT_MAX else 0
            counters["back"] = counters["back"] + 1 if angle_back < ANGLE_BACK_MIN else 0
            counters["valgus"] = counters["valgus"] + 1 if dist_knees < (dist_hips * 0.85) else 0

            # 5. WARNING CONFIGURATION
            
            if counters["depth"] >= FRAME_WINDOW: warning_texts.append("Depth not reached, go lower!")
            if counters["back"] >= FRAME_WINDOW: warning_texts.append("Back not straight, keep it upright!")
            if counters["valgus"] >= FRAME_WINDOW: warning_texts.append("Knees inward, keep them aligned with hips!")
            if is_jump: warning_texts.append("STABILITY: Detection jump!")

            color = (0, 0, 255) if warning_texts else (0, 255, 0)
            mp_drawing.draw_landmarks(layout[:, :w], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))

            # --- AFFICHAGE DU LAYOUT STYLE NOTIFICATION ---
        cv2.putText(layout, "NOTIFICATIONS", (w + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(layout, (w + 20, 55), (w + SIDEBAR_WIDTH - 20, 50), (100, 100, 100), 1)
            

        for i, text in enumerate(warning_texts):
            # Petit rectangle style bulle de notif
            rect_y = 80 + i * 70
            cv2.rectangle(layout, (w + 15, rect_y), (w + SIDEBAR_WIDTH - 15, rect_y + 50), (0, 0, 180), -1)
            cv2.putText(layout, text, (w + 25, rect_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the resulting frame
        cv2.imshow('Pose Estimation', layout)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()