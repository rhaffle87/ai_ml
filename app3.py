import cv2
import numpy as np
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import screen_brightness_control as sbc
import pythoncom

# Inisialisasi COM untuk Pycaw
pythoncom.CoInitialize()

# ==== Audio Setup (PyCaw) ====
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

# ==== Parameter Kalibrasi ====
CALIBRATION_FRAMES = 50

class HandGestureController(VideoTransformerBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Kalibrasi
        self.calibrating = True
        self.frame_count = 0
        self.vol_distances = []
        self.bright_distances = []

        # Rentang default
        self.vol_min, self.vol_max = 30, 200
        self.bright_min, self.bright_max = 30, 200

        # Statistik
        self.volume_level = 0
        self.brightness_level = 0

    def set_volume_from_distance(self, dist):
        volume = np.interp(dist, [self.vol_min, self.vol_max], [0.0, 1.0])
        volume = np.clip(volume, 0.0, 1.0)
        volume_interface.SetMasterVolumeLevelScalar(volume, None)
        self.volume_level = int(volume * 100)
        return self.volume_level

    def set_brightness_from_distance(self, dist):
        brightness = int(np.interp(dist, [self.bright_min, self.bright_max], [0, 100]))
        brightness = np.clip(brightness, 0, 100)
        sbc.set_brightness(brightness)
        self.brightness_level = brightness
        return self.brightness_level

    def reset_calibration(self):
        """Reset ulang kalibrasi"""
        self.calibrating = True
        self.frame_count = 0
        self.vol_distances = []
        self.bright_distances = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                h, w, _ = img.shape
                lm = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

                thumb_tip = lm[4]
                index_tip = lm[8]
                middle_tip = lm[12]

                vol_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
                bright_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))

                # ====== MODE KALIBRASI ======
                if self.calibrating:
                    self.vol_distances.append(vol_dist)
                    self.bright_distances.append(bright_dist)
                    self.frame_count += 1

                    cv2.putText(img, f"Calibrating... {self.frame_count}/{CALIBRATION_FRAMES}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    if self.frame_count >= CALIBRATION_FRAMES:
                        self.vol_min, self.vol_max = np.percentile(self.vol_distances, [5, 95])
                        self.bright_min, self.bright_max = np.percentile(self.bright_distances, [5, 95])
                        self.calibrating = False
                        print(f"Kalibrasi Volume: {self.vol_min:.1f} - {self.vol_max:.1f}")
                        print(f"Kalibrasi Brightness: {self.bright_min:.1f} - {self.bright_max:.1f}")

                else:
                    vol_level = self.set_volume_from_distance(vol_dist)
                    bri_level = self.set_brightness_from_distance(bright_dist)

                    cv2.putText(img, f"Volume: {vol_level}%", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(img, f"Brightness: {bri_level}%", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        return img


# ==== Streamlit UI ====
st.set_page_config(layout="wide")
st.title("🎛️ Gesture Control: Volume & Brightness")

col1, col2 = st.columns([3,1])

with col1:
    ctx = webrtc_streamer(
        key="gesture",
        video_transformer_factory=HandGestureController,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Real-Time Stats")

    if "volume_history" not in st.session_state:
        st.session_state["volume_history"] = []
    if "brightness_history" not in st.session_state:
        st.session_state["brightness_history"] = []

    if ctx.video_transformer:
        stats = ctx.video_transformer

        st.metric("🔊 Volume", f"{stats.volume_level}%")
        st.metric("💡 Brightness", f"{stats.brightness_level}%")

        st.write("Calibration:", "⏳ Ongoing" if stats.calibrating else "✅ Done")
        st.write(f"Volume Range: {stats.vol_min:.1f} - {stats.vol_max:.1f}")
        st.write(f"Brightness Range: {stats.bright_min:.1f} - {stats.bright_max:.1f}")

        # Tambahkan history untuk grafik
        if not stats.calibrating:
            st.session_state["volume_history"].append(stats.volume_level)
            st.session_state["brightness_history"].append(stats.brightness_level)

        # Grafik Line Chart
        st.line_chart({
            "Volume": st.session_state["volume_history"],
            "Brightness": st.session_state["brightness_history"]
        })

        # Tombol Recalibrate
        if st.button("🔄 Recalibrate"):
            stats.reset_calibration()
            st.session_state["volume_history"] = []
            st.session_state["brightness_history"] = []
            st.warning("Recalibration started... move your fingers naturally!")
