import streamlit as st
import requests
import pandas as pd
import os
import tempfile
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# -----------------------------
# Backend API URLs
# -----------------------------
API_URL_TEXT = "http://127.0.0.1:8000/predict/text"
API_URL_IMAGE = "http://127.0.0.1:8000/predict/image"
API_URL_VOICE = "http://127.0.0.1:8000/predict/voice"
USER_FILE = "users.csv"

# Ensure users.csv exists
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)

# -----------------------------
# Session state initialization
# -----------------------------
for key in ["page", "logged_in", "username", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key == "username" else [] if key == "chat_history" else "Home" if key == "page" else False

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Navigation")
if st.session_state.logged_in:
    choice = st.sidebar.radio(
        "Go to",
        ["Home", "Assistant", "Logout"],
        index=["Home", "Assistant", "Logout"].index(st.session_state.page) if st.session_state.page in ["Home", "Assistant"] else 0
    )
else:
    choice = st.sidebar.radio(
        "Go to",
        ["Home", "Login", "Register"],
        index=["Home", "Login", "Register"].index(st.session_state.page) if st.session_state.page in ["Home", "Login", "Register"] else 0
    )

if choice == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "Home"
    st.session_state.chat_history = []
    st.rerun()

st.session_state.page = choice

# -----------------------------
# Pages
# -----------------------------
if st.session_state.page == "Home":
    st.image("E:/unnamed.png", width=600)
    st.markdown("<h1 style='text-align:center; color:#4a148c;'>🤖 Smart Multimodal Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:18px; color:#00796b;'>Your AI-powered assistant for text, image & voice-based health insights.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Login"):
            st.session_state.page = "Login"
            st.rerun()
    with col2:
        if st.button("🆕 Register"):
            st.session_state.page = "Register"
            st.rerun()

elif st.session_state.page == "Login":
    st.title("🔑 Login")
    users = pd.read_csv(USER_FILE)
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Login"):
        if ((users['username'] == username) & (users['password'] == password)).any():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Welcome back, {username}!")
            st.session_state.page = "Assistant"
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

elif st.session_state.page == "Register":
    st.title("🆕 Register")
    username = st.text_input("👤 Choose a Username")
    password = st.text_input("🔒 Choose a Password", type="password")
    confirm = st.text_input("🔑 Confirm Password", type="password")
    if st.button("Register"):
        users = pd.read_csv(USER_FILE)
        if username.strip() == "" or password.strip() == "":
            st.warning("⚠ Username and password cannot be empty.")
        elif password != confirm:
            st.error("❌ Passwords do not match.")
        elif username in users['username'].values:
            st.error("⚠ Username already exists.")
        else:
            pd.DataFrame([[username, password]], columns=["username", "password"]).to_csv(USER_FILE, mode="a", header=False, index=False)
            st.success("🎉 Registration successful! Please login.")
            st.session_state.page = "Login"
            st.rerun()

# -----------------------------
# Assistant Page
# -----------------------------
elif st.session_state.page == "Assistant":
    st.title("🤖 Health Assistant Chat")
    st.markdown(f"👋 Hello {st.session_state.username}, tell me about your symptoms or upload a file!")

    # -----------------------------
    # Custom CSS for chat bubbles
    # -----------------------------
    st.markdown("""
        <style>
        .chat-bubble-user {
            background-color: #0078FF;
            color: white;
            padding: 10px 15px;
            border-radius: 12px;
            margin: 5px 0;
            text-align: right;
            width: fit-content;
            max-width: 80%;
            margin-left: auto;
        }
        .chat-bubble-assistant {
            background-color: #E5E5EA;
            color: black;
            padding: 10px 15px;
            border-radius: 12px;
            margin: 5px 0;
            text-align: left;
            width: fit-content;
            max-width: 80%;
            margin-right: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display chat history
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-assistant'>{msg['content']}</div>", unsafe_allow_html=True)

    # -----------------------------
    # Tabs for input types
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["💬 Text", "🖼 Image", "🎙 Voice"])

    # -----------------------------
    # Text Input
    # -----------------------------
    with tab1:
        user_text = st.text_area("Enter your symptoms:", height=100)
        if st.button("Send Text", key="text_btn"):
            if user_text.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_text})
                try:
                    response = requests.post(API_URL_TEXT, data={"text": user_text})
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get("prediction", "Unknown")
                        precautions = result.get("precautions", [])
                        reply = f"🧾 <b>Prediction:</b> {prediction}<br><br>💊 <b>Precautions:</b><br>" + "<br>".join([f"- {p}" for p in precautions])
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": "⚠ Error from backend"})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"⚠ Error: {e}"})
                st.rerun()

    # -----------------------------
    # Image Input
    # -----------------------------
    with tab2:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image and st.button("Send Image", key="img_btn"):
            st.session_state.chat_history.append({"role": "user", "content": "🖼 Sent an image"})
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_image.read())
                    tmp_path = tmp_file.name
                with open(tmp_path, "rb") as f:
                    response = requests.post(API_URL_IMAGE, files={"file": f})
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "Unknown")
                    precautions = result.get("precautions", [])
                    reply = f"🧾 <b>Prediction:</b> {prediction}<br><br>💊 <b>Precautions:</b><br>" + "<br>".join([f"- {p}" for p in precautions])
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "⚠ Error analyzing image"})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠ Error: {e}"})
            st.rerun()

    # -----------------------------
    # Voice Input (Record + Upload)
    # -----------------------------
    with tab3:
        st.subheader("🎤 Recorded Voice")
        recorded_audio = mic_recorder(start_prompt="🎙 Start Recording", stop_prompt="⏹ Stop Recording", key="recorder")
        if recorded_audio and st.button("Send Recorded Voice", key="rec_btn"):
            st.session_state.chat_history.append({"role": "user", "content": "🎤 Sent a recorded voice message"})
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
                    temp_webm.write(recorded_audio["bytes"])
                    webm_path = temp_webm.name
                wav_path = webm_path.replace(".webm", ".wav")
                AudioSegment.from_file(webm_path, format="webm").export(wav_path, format="wav")
                with open(wav_path, "rb") as f:
                    response = requests.post(API_URL_VOICE, files={"file": ("audio.wav", f, "audio/wav")})
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "Unknown")
                    confidence = result.get("confidence", 0)
                    precautions = result.get("precautions", [])
                    reply = f"🎤 <b>Prediction:</b> {prediction} (Confidence: {confidence:.2f})<br><br>💊 <b>Precautions:</b><br>" + "<br>".join([f"- {p}" for p in precautions])
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "⚠ Error from backend"})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠ Error: {e}"})
            st.rerun()

        st.subheader("🎤 Upload Voice File")
        uploaded_audio = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"], key="voice_upload")
        if uploaded_audio and st.button("Send Uploaded Voice", key="voice_btn"):
            st.session_state.chat_history.append({"role": "user", "content": "🎤 Sent an uploaded voice file"})
            try:
                ext = os.path.splitext(uploaded_audio.name)[-1].lower()
                tmp_ext = ".wav" if ext != ".wav" else ext
                with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_ext) as tmp_file:
                    tmp_file.write(uploaded_audio.read())
                    tmp_path = tmp_file.name
                if ext != ".wav":
                    wav_path = tmp_path.replace(ext, ".wav")
                    AudioSegment.from_file(tmp_path, format=ext.replace(".", "")).export(wav_path, format="wav")
                else:
                    wav_path = tmp_path
                with open(wav_path, "rb") as f:
                    response = requests.post(API_URL_VOICE, files={"file": ("audio.wav", f, "audio/wav")})
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "Unknown")
                    confidence = result.get("confidence", 0)
                    precautions = result.get("precautions", [])
                    reply = f"🎤 <b>Prediction:</b> {prediction} (Confidence: {confidence:.2f})<br><br>💊 <b>Precautions:</b><br>" + "<br>".join([f"- {p}" for p in precautions])
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "⚠ Error from backend"})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠ Error: {e}"})
            st.rerun()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🤖 AI Health Assistant — Built with FastAPI + Streamlit")