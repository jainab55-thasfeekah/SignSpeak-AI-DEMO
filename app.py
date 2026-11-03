import os
import time
import threading
import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="SignSpeak AI", page_icon="🤟", layout="wide")

st.markdown("""
<style>
.main {background: #f7f9fc;}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#172554,#2563eb); color:white;}
.hero h1 {color:white; margin:0;}
.hero p {color:#e0e7ff; margin:.35rem 0 0 0;}
.panel {background:white; border:1px solid #e5e7eb; padding:1rem; border-radius:14px;}
</style>
<div class="hero">
<h1>🤟 SignSpeak AI</h1>
<p>Real-time sign recognition • Text output • Speech support</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "sign_model.joblib"
LABEL_PATH = "sign_labels.txt"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

model = load_model()

with st.sidebar:
    st.header("Project controls")
    mode = st.radio("Choose mode", ["Live camera", "Demo mode"])
    st.caption("Live recognition requires a trained model file. Demo mode is clearly labeled and does not claim to recognize real signs.")
    st.divider()
    st.markdown("**Supported starter vocabulary**")
    st.write("HELLO · THANK YOU · YES · NO · HELP · WATER · PLEASE · SORRY")
    st.markdown("**Model status**")
    if model is not None:
        st.success("Trained model found")
    else:
        st.warning("No trained model yet")
    st.caption("Train one with the included collect_data.py and train_model.py scripts.")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Camera / gesture input")
    if mode == "Live camera":
        if model is None:
            st.info("Camera can be opened, but real predictions will only work after you collect labeled examples and train the model.")
        class SignProcessor(VideoProcessorBase):
            def __init__(self):
                self.lock = threading.Lock()
                self.frame = None
                self.prediction = "Waiting for hand..."
                self.confidence = 0.0
                self.last_pred_time = 0.0
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.55, min_tracking_confidence=0.5
                )
                self.drawer = mp.solutions.drawing_utils

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    self.drawer.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
                    coords = []
                    for p in hand.landmark:
                        coords.extend([p.x, p.y, p.z])
                    if model is not None:
                        try:
                            arr = np.array(coords, dtype=np.float32).reshape(1, -1)
                            pred = model.predict(arr)[0]
                            conf = 0.0
                            if hasattr(model, "predict_proba"):
                                conf = float(np.max(model.predict_proba(arr)))
                            with self.lock:
                                self.prediction = str(pred)
                                self.confidence = conf
                        except Exception:
                            with self.lock:
                                self.prediction = "Prediction error"
                                self.confidence = 0.0
                    else:
                        cv2.putText(img, "Hand detected - train model for labels", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                else:
                    with self.lock:
                        self.prediction = "No hand detected"
                        self.confidence = 0.0
                with self.lock:
                    label = self.prediction
                    conf = self.confidence
                cv2.rectangle(img, (8, img.shape[0]-48), (min(img.shape[1]-8, 420), img.shape[0]-8), (20, 30, 55), -1)
                cv2.putText(img, f"{label}  {conf:.0%}", (18, img.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="signspeak-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx and ctx.video_processor:
            with ctx.video_processor.lock:
                current_prediction = ctx.video_processor.prediction
                current_confidence = ctx.video_processor.confidence
            st.metric("Current prediction", current_prediction, f"Confidence: {current_confidence:.0%}")
            if current_prediction not in ["No hand detected", "Waiting for hand...", "Prediction error"] and model is not None:
                if st.button("Add prediction to sentence"):
                    st.session_state["sentence"] = (st.session_state.get("sentence", "") + " " + current_prediction).strip()
    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write("Use this mode to demonstrate the user interface when no trained model is ready.")
        demo_sign = st.selectbox("Choose a sample sign", ["HELLO", "THANK YOU", "YES", "NO", "HELP", "WATER", "PLEASE", "SORRY"])
        st.caption("Demo selection is simulated input, not camera recognition.")
        if st.button("Show sample translation", type="primary"):
            st.session_state["demo_prediction"] = demo_sign
        if st.session_state.get("demo_prediction"):
            st.success(f"Sample output: {st.session_state['demo_prediction']}")
            if st.button("Add sample to sentence"):
                st.session_state["sentence"] = (st.session_state.get("sentence", "") + " " + st.session_state["demo_prediction"]).strip()
        st.markdown('</div>', unsafe_allow_html=True)

def clear_sentence():
    st.session_state["sentence"] = ""

with right:
    st.subheader("Text and speech output")
    prediction = st.session_state.get("demo_prediction", "") if mode == "Demo mode" else ""
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Sentence builder**")
    st.session_state.setdefault("sentence", "")
    st.text_area("Recognized text", key="sentence", height=150, placeholder="Recognized words will appear here...")
    b1, b2 = st.columns(2)
    with b1:
        st.button("Clear text", use_container_width=True, on_click=clear_sentence)
    with b2:
        speak = st.button("🔊 Speak text", use_container_width=True, type="primary")
    if speak:
        text_to_speak = st.session_state.get("sentence", "").strip()
        if text_to_speak:
            # Browser speech synthesis is used so no server audio device is required.
            import streamlit.components.v1 as components
            safe_text = text_to_speak.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            components.html(f"""
            <button onclick="const u=new SpeechSynthesisUtterance('{safe_text}'); u.lang='en-IN'; speechSynthesis.cancel(); speechSynthesis.speak(u);"
            style="padding:10px 16px;border:0;border-radius:10px;background:#2563eb;color:white;font-weight:600;cursor:pointer;">
            Play speech in browser</button>
            <script>
            const u = new SpeechSynthesisUtterance('{safe_text}');
            u.lang = 'en-IN';
            speechSynthesis.cancel();
            speechSynthesis.speak(u);
            </script>
            """, height=70)
        else:
            st.warning("Add some text before speaking.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Prototype checklist**")
    st.write("✅ Text output")
    st.write("✅ Sentence builder and clear")
    st.write("✅ Browser speech output")
    st.write("✅ Webcam hand landmarks")
    st.write("⚠️ Real translation depends on a model trained on labeled signs")
    st.caption("This is an assistive prototype, not a replacement for a qualified interpreter.")
