import os
import cv2
import mediapipe as mp
import numpy as np



def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

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

    print (f"Processing video: {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # mediapipe requires RGB images, so convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find pose landmarks
        results = pose.process(rgb_frame)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y

            angle_knee = calculate_angle(hip, knee, ankle)

            # ANGLE DISPLAY
            cv2.putText(frame, f"Knee Angle: {int(angle_knee)} deg",
                        tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- DEPTH ESTIMATION ---
            if angle_knee < 90:
                cv2.putText(frame, "VALID SQUAT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
            
            # Example: Extracting the coordinates of the left knee (landmark index 25)
            #we multiply by the frame dimensions to get pixel coordinates
            h, w, _ = frame.shape
            knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            knee_l_x = int(knee_l.x * w)
            knee_l_y = int(knee_l.y * h)


        # Display the resulting frame
        cv2.imshow('Pose Estimation', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()