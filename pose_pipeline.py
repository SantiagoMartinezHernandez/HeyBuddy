"""
Pose Pipeline — Phase 1
=======================
Detects body landmarks, computes joint angles, tracks movement amplitude.

Usage:
  python pose_pipeline.py                           # webcam
  python pose_pipeline.py Media/video.mp4           # video file (display only)
  python pose_pipeline.py Media/video.mp4 out.mp4   # video file + save output
  python pose_pipeline.py Media/video.mp4 --no-show # process silently
"""

import os       # file path utilities (check if file exists, split extension, etc.)
import sys      # access command-line arguments (sys.argv)
import json     # save results as a JSON file
import time     # get current time in milliseconds (used for webcam timestamps)

import cv2      # OpenCV — reads videos, draws on frames, shows windows
import numpy as np                          # math on arrays (vectors, dot product...)
import mediapipe as mp                      # Google's pose detection library
from mediapipe.tasks import python          # MediaPipe Tasks API base options
from mediapipe.tasks.python import vision  # MediaPipe vision tasks (PoseLandmarker)


# =============================================================================
# LANDMARK NAMES
# =============================================================================
# MediaPipe detects 33 body points (called "landmarks") on a person.
# Each landmark has an index from 0 to 32.
# This list maps each index to a human-readable name so we can refer to
# "left_knee" instead of index 25.
LM_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Build a reverse lookup: name → index.
# e.g. LM["left_knee"] == 25
# This lets us write readable code like LM["left_knee"] instead of the raw number 25.
LM = {name: i for i, name in enumerate(LM_NAMES)}


# =============================================================================
# JOINTS TO MEASURE
# =============================================================================
# To measure a joint angle we need 3 points: A — B — C
# where B is the joint vertex (the hinge) and A, C are the two bones around it.
# Example for the left knee:
#   A = left hip, B = left knee, C = left ankle
#   → the angle tells us how bent the knee is (180° = straight leg, ~90° = right-angle bend)
#
# Format: "joint_name": (index_A, index_B_vertex, index_C)
JOINTS = {
    "left_knee":   (LM["left_hip"],       LM["left_knee"],   LM["left_ankle"]),
    "right_knee":  (LM["right_hip"],      LM["right_knee"],  LM["right_ankle"]),
    "left_elbow":  (LM["left_shoulder"],  LM["left_elbow"],  LM["left_wrist"]),
    "right_elbow": (LM["right_shoulder"], LM["right_elbow"], LM["right_wrist"]),
    "left_hip":    (LM["left_shoulder"],  LM["left_hip"],    LM["left_knee"]),
    "right_hip":   (LM["right_shoulder"], LM["right_hip"],   LM["right_knee"]),
}

# =============================================================================
# SKELETON CONNECTIONS
# =============================================================================
# These pairs of landmark indices define which points to connect with a line
# when drawing the skeleton on the frame.
# e.g. (11, 13) means "draw a line from left_shoulder (11) to left_elbow (13)"
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),           # shoulders & arms
    (11, 23), (12, 24), (23, 24),                                # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                      # legs
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32), # feet
]

# The joint vertex points (B in A-B-C) — we'll draw these larger and in green
# so they stand out as the "measured" joints.
INTEREST_IDS = {v for _, v, _ in JOINTS.values()}

# Face landmark indices (nose, eyes, ears, mouth) — indices 0 to 10.
# We skip drawing these to keep the overlay focused on the body.
FACE_IDS = set(range(11))


# =============================================================================
# ANGLE CALCULATION
# =============================================================================

def calc_3d_angle(a, b, c) -> float:
    """
    Calculates the angle (in degrees) at point B, between the segments BA and BC.

    Think of it like this: if you hold your arm out straight, the elbow angle
    is ~180°. If you bend it fully, it's ~30°.

    How it works (vector math):
      1. Build two vectors from B: one pointing to A, one pointing to C.
      2. Use the dot product formula:  cos(angle) = (BA · BC) / (|BA| * |BC|)
      3. Take the arccos to get the angle, convert from radians to degrees.

    The 1e-9 in the denominator avoids division by zero if two points overlap.
    np.clip keeps the cosine in [-1, 1] to avoid math errors from floating-point noise.
    """
    # Convert landmark dicts to numpy arrays for vector math
    a = np.array([a["x"], a["y"], a["z"]])
    b = np.array([b["x"], b["y"], b["z"]])
    c = np.array([c["x"], c["y"], c["z"]])

    # Vectors from the vertex B toward A and C
    ba = a - b
    bc = c - b

    # Cosine of the angle between BA and BC
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)

    # arccos gives the angle in radians, np.degrees converts to degrees
    return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def draw_skeleton(frame, lm_list):
    """
    Draws the skeleton (bones + joint dots) on the frame.

    - White lines connect landmark pairs (the "bones").
    - Green filled circles mark the joint vertices we are measuring.
    - Orange dots mark all other visible landmarks.

    Landmarks with visibility < 0.5 are skipped (MediaPipe is not confident
    enough about their position, so we don't draw them).
    """
    h, w, _ = frame.shape  # frame dimensions in pixels

    # Build a dict: landmark_id → pixel position (x, y)
    # MediaPipe gives coordinates as fractions of the frame (0.0 to 1.0),
    # so we multiply by width/height to get actual pixel positions.
    coords = {
        lm["id"]: (int(lm["x"] * w), int(lm["y"] * h))
        for lm in lm_list if lm["visibility"] > 0.5
    }

    # Draw bone connections (white lines)
    for a, b in CONNECTIONS:
        if a in coords and b in coords:  # only draw if both endpoints are visible
            cv2.line(frame, coords[a], coords[b], (255, 255, 255), 2)

    # Draw landmark dots
    for lm_id, pt in coords.items():
        if lm_id in FACE_IDS:
            continue  # skip face points entirely
        if lm_id in INTEREST_IDS:
            # Joint we measure: large green dot with white border
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)   # filled green circle
            cv2.circle(frame, pt, 8, (255, 255, 255), 2) # white ring around it
        else:
            # Other landmarks: small orange dot
            cv2.circle(frame, pt, 4, (255, 100, 0), -1)


def draw_angles(frame, lm_list, angles):
    """
    Prints each joint angle as text next to the corresponding joint on the frame.

    For example, "142°" appears next to the left knee landmark.
    """
    h, w, _ = frame.shape

    # Pixel position of each visible landmark
    coords = {
        lm["id"]: (int(lm["x"] * w), int(lm["y"] * h))
        for lm in lm_list if lm["visibility"] > 0.5
    }

    # Map joint name → its vertex landmark index
    # e.g. "left_knee" → 25
    vertex_ids = {name: v for name, (_, v, _) in JOINTS.items()}

    for joint_name, angle in angles.items():
        vid = vertex_ids[joint_name]
        if vid in coords:
            x, y = coords[vid]
            # Draw the angle value slightly to the right of the joint dot
            cv2.putText(frame, f"{int(angle)}°", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


# =============================================================================
# AMPLITUDE TRACKER
# =============================================================================

class AmplitudeTracker:
    """
    Keeps track of how much each joint has moved during the whole session.

    For each joint it records:
      - the smallest angle seen (most bent position)
      - the largest angle seen (most extended position)
      - the difference = "amplitude" = total range of motion

    Example: if the left knee went from 45° (bent) to 170° (straight),
    the amplitude is 125°.
    """

    def __init__(self):
        self._min: dict = {}  # minimum angle seen per joint
        self._max: dict = {}  # maximum angle seen per joint

    def update(self, angles: dict):
        """Call this every frame with the current joint angles."""
        for joint, angle in angles.items():
            if joint not in self._min:
                # First time we see this joint — initialise both min and max
                self._min[joint] = angle
                self._max[joint] = angle
            else:
                # Update if this frame has a more extreme angle
                self._min[joint] = min(self._min[joint], angle)
                self._max[joint] = max(self._max[joint], angle)

    def summary(self) -> dict:
        """Returns a dict with min, max, and amplitude for every tracked joint."""
        return {
            joint: {
                "min_deg": round(self._min[joint], 1),
                "max_deg": round(self._max[joint], 1),
                "amplitude_deg": round(self._max[joint] - self._min[joint], 1),
            }
            for joint in self._min
        }


# =============================================================================
# FRAME PROCESSOR
# =============================================================================

def process_frame(landmarker, frame, timestamp_ms):
    """
    Runs pose detection on a single video frame and annotates it.

    Steps:
      1. Convert the frame from BGR (OpenCV default) to RGB (MediaPipe expects RGB).
      2. Wrap it in a MediaPipe Image object.
      3. Run the landmarker — it returns the 33 landmark positions.
      4. If no person is detected, return the frame unchanged.
      5. Build a list of landmark dicts with position and visibility info.
      6. Compute angles for each configured joint.
      7. Draw the skeleton and angle labels onto the frame.

    Returns:
      annotated_frame  — the frame with drawings on it
      lm_list          — list of landmark dicts (empty if no pose detected)
      angles           — dict of joint_name → angle in degrees (empty if no pose)
    """
    # Step 1 & 2: convert colour space and wrap in MediaPipe image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Step 3: run the pose landmarker
    # timestamp_ms is required in VIDEO mode so MediaPipe can smooth motion over time
    res = landmarker.detect_for_video(mp_img, timestamp_ms)

    # Step 4: if nothing detected, return the original frame untouched
    if not res.pose_landmarks:
        return frame, [], {}

    # Step 5: build a list of landmark dicts
    # res.pose_landmarks[0] = first detected person (index 0)
    # Each lm has .x, .y, .z (fractions of frame size) and .visibility (0–1 confidence)
    lm_list = [
        {
            "id": i,
            "name": LM_NAMES[i],
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),  # depth estimate (less reliable than x/y)
            "visibility": round(lm.visibility, 4) if hasattr(lm, "visibility") else 1.0,
        }
        for i, lm in enumerate(res.pose_landmarks[0])
    ]

    # Quick-access dict: landmark_id → landmark dict
    by_id = {lm["id"]: lm for lm in lm_list}

    # Step 6: compute angles for each joint
    angles = {}
    for joint_name, (a_id, b_id, c_id) in JOINTS.items():
        a, b, c = by_id.get(a_id), by_id.get(b_id), by_id.get(c_id)
        # Only compute if all three landmarks exist AND the vertex is visible enough
        if a and b and c and b["visibility"] > 0.3:
            angles[joint_name] = calc_3d_angle(a, b, c)

    # Step 7: draw skeleton and angle labels on the frame
    draw_skeleton(frame, lm_list)
    draw_angles(frame, lm_list, angles)

    return frame, lm_list, angles


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(
    source=0,
    output_path: str = None,
    model_path: str = "pose_landmarker_lite.task",
    show: bool = True,
):
    """
    Runs the full pose pipeline on a video source.

    Args:
        source:      0 = webcam, or a string path to a video file.
        output_path: If given, saves the annotated video to this path.
        model_path:  Path to the MediaPipe .task model file.
        show:        If True, opens a window to display the video live.
    """

    # --- Load the MediaPipe model ---
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # VIDEO mode uses timestamps to smooth results
        min_pose_detection_confidence=0.5,      # minimum confidence to start tracking a person
        min_tracking_confidence=0.5,            # minimum confidence to keep tracking
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # --- Open the video source ---
    # cv2.VideoCapture(0) opens the default webcam.
    # cv2.VideoCapture("path/to/file.mp4") opens a video file.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Read video metadata
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0   # fallback to 30 if unknown
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0 for webcam
    is_file = isinstance(source, str)

    # --- Set up video writer (optional) ---
    writer = None
    if output_path:
        # mp4v is a widely compatible codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = AmplitudeTracker()
    frame_idx = 0  # counts how many frames we have processed

    print(f"Source : {'webcam' if source == 0 else source}")
    print(f"Output : {output_path or 'none (display only)'}")
    if total > 0:
        print(f"Frames : {total}")
    if is_file:
        print("Press P to pause/resume, Q to quit.\n")
    else:
        print("Press Q to quit.\n")

    # --- Main loop: read and process frames one by one ---
    while cap.isOpened():
        ret, frame = cap.read()  # ret = True if a frame was successfully read
        if not ret:
            break  # end of file or camera disconnected

        # Timestamps must always increase for MediaPipe VIDEO mode.
        # For files: use the actual video position in milliseconds.
        # For webcam: use the system clock.
        timestamp_ms = (
            int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if is_file
            else int(time.time() * 1000)
        )

        # Run pose detection and draw results on the frame
        annotated, lm_list, angles = process_frame(landmarker, frame, timestamp_ms)

        # Update the min/max angle tracker
        tracker.update(angles)

        # Burn a frame counter into the top-left corner of the video
        label = f"Frame {frame_idx + 1}" + (f"/{total}" if total > 0 else "")
        cv2.putText(annotated, label, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the annotated frame to the output file (if requested)
        if writer:
            writer.write(annotated)

        # Show the frame in a window (if requested)
        if show:
            cv2.imshow("Pose Pipeline", annotated)
            # waitKey(1) waits 1 ms for a keypress; & 0xFF isolates the key code
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Pause only makes sense for video files (not live webcam)
            if key == ord("p") and is_file:
                print("Paused — press P to resume, Q to quit.")
                # Keep showing the current frame until P or Q is pressed again
                while True:
                    k = cv2.waitKey(50) & 0xFF  # check every 50 ms
                    if k == ord("p"):
                        print("Resumed.")
                        break
                    if k == ord("q"):
                        cap.release()           # force the outer loop to stop
                        if writer:
                            writer.release()
                        landmarker.close()
                        cv2.destroyAllWindows()
                        raise SystemExit(0)

        frame_idx += 1

        # Print progress every 50 frames so you can see it is running
        if frame_idx % 50 == 0:
            suffix = f"/{total}" if total > 0 else ""
            print(f"  {frame_idx}{suffix} frames processed")

    # --- Cleanup ---
    cap.release()       # release the video file / webcam
    if writer:
        writer.release()
    landmarker.close()  # free the MediaPipe model from memory
    if show:
        cv2.destroyAllWindows()

    # --- Print amplitude summary ---
    summary = tracker.summary()
    print("\n-- Amplitude Summary " + "-" * 40)
    for joint, stats in summary.items():
        print(
            f"  {joint:20s}  "
            f"min={stats['min_deg']:6.1f}  "
            f"max={stats['max_deg']:6.1f}  "
            f"amplitude={stats['amplitude_deg']:6.1f} deg"
        )

    # --- Save JSON summary ---
    # Determine where to save it: next to the output video, or next to the input file.
    json_path = None
    if output_path:
        json_path = os.path.splitext(output_path)[0] + "_summary.json"
    elif is_file:
        json_path = os.path.splitext(source)[0] + "_summary.json"

    if json_path:
        with open(json_path, "w") as f:
            json.dump({"frames_processed": frame_idx, "joints": summary}, f, indent=2)
        print(f"\nSummary saved: {json_path}")

    return summary


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================
# This block only runs when you execute the script directly:
#   python pose_pipeline.py ...
# It does NOT run when another script imports this file (e.g. import pose_pipeline).

if __name__ == "__main__":
    args = sys.argv[1:]  # sys.argv[0] is the script name itself, skip it

    # Check for the --no-show flag anywhere in the arguments
    show = "--no-show" not in args
    args = [a for a in args if a != "--no-show"]  # remove the flag from the list

    source = 0    # default: webcam
    output = None # default: don't save

    if len(args) >= 1:
        src = args[0]
        # If the argument is a plain number (e.g. "0"), treat it as a camera index.
        # Otherwise treat it as a file path.
        source = int(src) if src.isdigit() else src

    if len(args) >= 2:
        output = args[1]

    run(source=source, output_path=output, show=show)
