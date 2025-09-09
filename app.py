import cv2
import numpy as np
import math
import streamlit as st

# === Mediapipe ===
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === Audio Control (PyCaw) ===
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# === Brightness Control ===
import screen_brightness_control as sbc

# --------------------
# Setup Streamlit UI
# --------------------
st.set_page_config(page_title="Gesture Control", layout="wide")
st.title("🖐️ Gesture Control: Volume & Brightness")

FRAME_WINDOW = st.image([])

# --------------------
# Setup PyCaw (Audio)
# --------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

def set_volume_from_distance(dist):
    MIN_DIST, MAX_DIST = 30, 200
    volume = np.interp(dist, [MIN_DIST, MAX_DIST], [0.0, 1.0])
    volume_interface.SetMasterVolumeLevelScalar(volume, None)
    return int(volume*100)

def set_brightness_from_distance(dist):
    MIN_DIST, MAX_DIST = 20, 150
    brightness = int(np.interp(dist, [MIN_DIST, MAX_DIST], [0, 100]))
    sbc.set_brightness(brightness)
    return brightness

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_coords(hand_landmarks, image_w, image_h):
    return [(int(lm.x * image_w), int(lm.y * image_h)) for lm in hand_landmarks]

# --------------------
# Mediapipe Hand Model
# --------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def draw_and_control(image, hand_landmarks):
    image_h, image_w, _ = image.shape
    annotated = image.copy()
    coords = get_coords(hand_landmarks, image_w, image_h)

    # Draw skeleton
    for connection in HAND_CONNECTIONS:
        start = coords[connection[0]]
        end = coords[connection[1]]
        cv2.line(annotated, start, end, (0, 0, 255), 2)
    for (x, y) in coords:
        cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

    # Kontrol Volume (Thumb–Index)
    thumb_tip = coords[4]
    index_tip = coords[8]
    dist_vol = distance(thumb_tip, index_tip)
    cv2.line(annotated, thumb_tip, index_tip, (255, 255, 0), 3)
    vol_level = set_volume_from_distance(dist_vol)
    cv2.putText(annotated, f"Vol:{vol_level}%", (thumb_tip[0], thumb_tip[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Kontrol Brightness (Index–Middle)
    middle_tip = coords[12]
    dist_bri = distance(index_tip, middle_tip)
    cv2.line(annotated, index_tip, middle_tip, (0, 255, 255), 3)
    bright_level = set_brightness_from_distance(dist_bri)
    cv2.putText(annotated, f"Bri:{bright_level}%", (index_tip[0], index_tip[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return annotated

# --------------------
# Streamlit Loop
# --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("❌ Kamera tidak terbaca")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    annotated_frame = frame.copy()
    for hand_landmarks in result.hand_landmarks:
        annotated_frame = draw_and_control(annotated_frame, hand_landmarks)

    FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()