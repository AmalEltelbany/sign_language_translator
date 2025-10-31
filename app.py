import os
import time
import base64
import json
import numpy as np
import cv2

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("SIGN_MODEL_PATH",  "model/cv_model.hdf5")
LABELS_PATH = os.getenv("SIGN_LABELS_PATH", "model/names")

from stos.sign_to_speech.sign_to_speech import SignToSpeech
from stos.speech_to_sign.speech_to_sign import SpeechToSignTranslator

sign_translator   = SignToSentenceTranslator(MODEL_PATH, LABELS_PATH)
speech_translator = SpeechToSignTranslator()

# ─── FLASK + SOCKET.IO SETUP ─────────────────────────────────────────────────
# tell Flask where to find templates + static files
app = Flask(__name__, static_folder="static", template_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

# ─── ROUTE TO SERVE YOUR HTML CLIENT ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─── IMAGE DECODE HELPER ─────────────────────────────────────────────────────
def decode_base64_image(data_uri: str):
    b64 = data_uri.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ─── MEETING MANAGEMENT ───────────────────────────────────────────────────────
@socketio.on("create_meeting")
def on_create_meeting(payload):
    room = payload["meeting_id"]
    join_room(room)
    emit("meeting_created", {"meeting_id": room})

@socketio.on("join_meeting")
def on_join_meeting(payload):
    room = payload["meeting_id"]
    user = payload.get("user", "anonymous")
    join_room(room)
    emit("user_joined", {"user": user}, room=room)

@socketio.on("leave_meeting")
def on_leave_meeting(payload):
    room = payload["meeting_id"]
    leave_room(room)
    emit("user_left", {"user": payload.get("user")}, room=room)

# ─── LIVE SIGN → TEXT ────────────────────────────────────────────────────────
_last_process = {}
@socketio.on("live_sign_frame")
def on_live_sign_frame(payload):
    room = payload["meeting_id"]
    now  = time.time()
    last = _last_process.get(room, 0.0)
    if now - last < 0.1:
        return
    _last_process[room] = now

    img      = decode_base64_image(payload["frame"])
    sentence = sign_translator.translate(img)
    emit("translation_sign2text", {"text": sentence}, room=room)

# ─── LIVE TEXT → SIGN ────────────────────────────────────────────────────────
@socketio.on("live_speech_text")
def on_live_speech_text(payload):
    room = payload["meeting_id"]
    text = payload["text"]
    video_data_uri = speech_translator.text_to_sign_video(text)
    emit("translation_text2sign", {"video": video_data_uri}, room=room)

# ─── OPTIONAL AUDIO CHUNKS ────────────────────────────────────────────────────
@socketio.on("live_audio_chunk")
def on_live_audio_chunk(payload):
    pass

# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
