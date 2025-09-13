import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ====== Global Variables ======
CALIBRATION_FRAMES = 50
vol_min, vol_max = 30, 200
bright_min, bright_max = 30, 200
calibrating = True
frame_count = 0
vol_distances = []
bright_distances = []

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ====== Fungsi Set Volume & Brightness ======
def set_volume_from_distance(dist):
    norm = np.interp(dist, [vol_min, vol_max], [0, 100])
    return int(np.clip(norm, 0, 100))

def set_brightness_from_distance(dist):
    norm = np.interp(dist, [bright_min, bright_max], [0, 100])
    return int(np.clip(norm, 0, 100))

# ====== Video Transformer ======
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7,
                                    min_tracking_confidence=0.7)

    def transform(self, frame):
        global calibrating, frame_count, vol_distances, bright_distances
        global vol_min, vol_max, bright_min, bright_max

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        vol_level, bright_level = None, None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                lm = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

                thumb_tip, index_tip, middle_tip = lm[4], lm[8], lm[12]
                vol_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
                bright_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))

                if calibrating:
                    vol_distances.append(vol_dist)
                    bright_distances.append(bright_dist)
                    frame_count += 1
                    cv2.putText(img, f"Calibrating... {frame_count}/{CALIBRATION_FRAMES}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if frame_count >= CALIBRATION_FRAMES:
                        vol_min, vol_max = np.percentile(vol_distances, [5, 95])
                        bright_min, bright_max = np.percentile(bright_distances, [5, 95])
                        calibrating = False
                else:
                    vol_level = set_volume_from_distance(vol_dist)
                    bright_level = set_brightness_from_distance(bright_dist)

                    cv2.putText(img, f"Volume: {vol_level}%", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Brightness: {bright_level}%", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Update stats ke Streamlit
        if not calibrating and vol_level is not None and bright_level is not None:
            st.session_state["volume"] = vol_level
            st.session_state["brightness"] = bright_level
            st.session_state["vol_history"].append(vol_level)
            st.session_state["bright_history"].append(bright_level)

        return img

# ====== Streamlit UI ======
st.set_page_config(layout="wide")
st.title("🎥 Gesture Control: Volume & Brightness")

# Init Session State
if "volume" not in st.session_state:
    st.session_state["volume"] = 0
if "brightness" not in st.session_state:
    st.session_state["brightness"] = 0
if "vol_history" not in st.session_state:
    st.session_state["vol_history"] = []
if "bright_history" not in st.session_state:
    st.session_state["bright_history"] = []

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(key="gesture", video_transformer_factory=HandGestureTransformer)

with col2:
    st.subheader("📊 Realtime Stats")
    st.metric("Volume", f"{st.session_state['volume']}%")
    st.metric("Brightness", f"{st.session_state['brightness']}%")

    # Tombol Recalibrate
    if st.button("🔄 Recalibrate"):
        calibrating = True
        frame_count = 0
        vol_distances = []
        bright_distances = []
        st.session_state["vol_history"] = []
        st.session_state["bright_history"] = []
        st.warning("Recalibration started... move your fingers naturally!")

    # Grafik Line Chart
    st.line_chart({
        "Volume": st.session_state["vol_history"],
        "Brightness": st.session_state["bright_history"]
    })
    st.write(f"Volume Range: {vol_min:.1f} - {vol_max:.1f}")
    st.write(f"Brightness Range: {bright_min:.1f} - {bright_max:.1f}")
# ====== End of File ======