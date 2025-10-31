import os
import uuid
import time
import logging
import tempfile
import traceback
import json
import asyncio
import threading
from datetime import datetime
from collections import deque
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from waitress import serve
import speech_recognition as sr
import mediapipe as mp

# TensorFlow log suppression
os.environ['GLOG_minloglevel'] = '2'
import tensorflow as tf

# Custom translation classes (fallback to mock if not available)
try:
    from stos.sign_to_speech.model import Model
    from stos.sign_to_speech.parser import Parser
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logging.warning("Sign-to-speech model not available. Using mock implementation.")

# Flask and SocketIO setup
app = Flask(__name__, static_folder='static', template_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_meeting_translator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_meeting_translator")

# Global variables for meeting management
active_meetings = {}
user_sessions = {}

# MediaPipe setup for optimized pose detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class MeetingRoom:
    def __init__(self, room_id):
        self.room_id = room_id
        self.participants = {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.translation_buffer = deque(maxlen=10)
        self.sign_processor = OptimizedSignProcessor()
        
    def add_participant(self, user_id, socket_id, username="Anonymous"):
        self.participants[user_id] = {
            'socket_id': socket_id,
            'username': username,
            'joined_at': datetime.now(),
            'is_speaking': False,
            'is_signing': False
        }
        self.last_activity = datetime.now()
        logger.info(f"User {username} ({user_id}) joined room {self.room_id}")
        
    def remove_participant(self, user_id):
        if user_id in self.participants:
            username = self.participants[user_id].get('username', 'Unknown')
            del self.participants[user_id]
            self.last_activity = datetime.now()
            logger.info(f"User {username} ({user_id}) left room {self.room_id}")
            
    def get_participant_count(self):
        return len(self.participants)
        
    def update_activity(self):
        self.last_activity = datetime.now()

class OptimizedSignProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_process_time = 0
        self.keypoint_buffer = deque(maxlen=20)  # Buffer for sequence processing
        
        # Pre-computed sign patterns for common words
        self.sign_patterns = self._load_sign_patterns()
        
    def _load_sign_patterns(self):
        """Load pre-computed sign patterns for fast recognition"""
        # This would normally load from a trained model
        # For now, return mock patterns
        return {
            'hello': {'confidence': 0.8, 'pattern': 'hand_wave'},
            'thanks': {'confidence': 0.75, 'pattern': 'hand_to_chin'},
            'yes': {'confidence': 0.9, 'pattern': 'nod_fist'},
            'no': {'confidence': 0.85, 'pattern': 'shake_finger'},
            'help': {'confidence': 0.7, 'pattern': 'hand_up'},
            'okay': {'confidence': 0.8, 'pattern': 'ok_gesture'},
            'good': {'confidence': 0.75, 'pattern': 'thumbs_up'},
            'sorry': {'confidence': 0.7, 'pattern': 'hand_circle_chest'}
        }
    
    def process_frame_fast(self, frame_data):
        """Optimized frame processing with reduced latency"""
        current_time = time.time()
        
        # Rate limiting: Process at most 10 FPS to reduce load
        if current_time - self.last_process_time < 0.1:
            return None
            
        self.last_process_time = current_time
        
        try:
            # Convert base64 to image
            if isinstance(frame_data, str):
                import base64
                frame_bytes = base64.b64decode(frame_data.split(',')[1])
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
                
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            
            # Extract keypoints
            keypoints = self._extract_keypoints(hand_results, pose_results)
            
            if keypoints:
                self.keypoint_buffer.append(keypoints)
                
                # Try to recognize sign if we have enough frames
                if len(self.keypoint_buffer) >= 5:
                    recognized_word = self._recognize_sign_pattern()
                    if recognized_word:
                        return {
                            'word': recognized_word['word'],
                            'confidence': recognized_word['confidence'],
                            'timestamp': current_time
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return None
    
    def _extract_keypoints(self, hand_results, pose_results):
        """Extract relevant keypoints from MediaPipe results"""
        keypoints = {}
        
        # Hand keypoints
        if hand_results.multi_hand_landmarks:
            hands_data = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                hands_data.append(hand_points)
            keypoints['hands'] = hands_data
        
        # Pose keypoints (upper body only for efficiency)
        if pose_results.pose_landmarks:
            pose_points = []
            # Only extract upper body landmarks (0-16)
            for i in range(17):
                landmark = pose_results.pose_landmarks.landmark[i]
                pose_points.extend([landmark.x, landmark.y, landmark.z])
            keypoints['pose'] = pose_points
            
        return keypoints if keypoints else None
    
    def _recognize_sign_pattern(self):
        """Fast sign recognition using pattern matching"""
        if len(self.keypoint_buffer) < 5:
            return None
            
        # Simple pattern matching (would be replaced with actual ML model)
        # For demo purposes, randomly select from available patterns
        import random
        
        # Simulate processing time
        if random.random() > 0.7:  # 30% chance of recognition
            word = random.choice(list(self.sign_patterns.keys()))
            return {
                'word': word,
                'confidence': self.sign_patterns[word]['confidence']
            }
        
        return None

class FastSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5  # Shorter pause for faster response
        
    def process_audio_chunk(self, audio_data):
        """Process audio chunk for speech recognition"""
        try:
            # Convert audio data to AudioData object
            audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            
            # Quick recognition with timeout
            text = self.recognizer.recognize_google(audio, timeout=2)
            return text.lower()
            
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None

# Initialize global processors
sign_processor = OptimizedSignProcessor()
speech_recognizer = FastSpeechRecognizer()

# SocketIO Event Handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    user_sessions[request.sid] = {
        'connected_at': datetime.now(),
        'room_id': None,
        'user_id': None
    }

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    
    # Remove from room if in one
    session = user_sessions.get(request.sid)
    if session and session['room_id'] and session['user_id']:
        room_id = session['room_id']
        user_id = session['user_id']
        
        if room_id in active_meetings:
            active_meetings[room_id].remove_participant(user_id)
            
            # Notify other participants
            emit_to_room(room_id, 'participant_left', {
                'user_id': user_id,
                'participant_count': active_meetings[room_id].get_participant_count()
            })
            
            # Remove room if empty
            if active_meetings[room_id].get_participant_count() == 0:
                del active_meetings[room_id]
                logger.info(f"Room {room_id} deleted (empty)")
    
    # Clean up session
    if request.sid in user_sessions:
        del user_sessions[request.sid]

@socketio.on('join_meeting')
def handle_join_meeting(data):
    room_id = data.get('room_id')
    user_id = data.get('user_id') or str(uuid.uuid4())
    username = data.get('username', 'Anonymous')
    
    if not room_id:
        emit('error', {'message': 'Room ID required'})
        return
    
    # Create room if it doesn't exist
    if room_id not in active_meetings:
        active_meetings[room_id] = MeetingRoom(room_id)
        logger.info(f"Created new meeting room: {room_id}")
    
    # Join the room
    join_room(room_id)
    meeting = active_meetings[room_id]
    meeting.add_participant(user_id, request.sid, username)
    
    # Update session
    user_sessions[request.sid]['room_id'] = room_id
    user_sessions[request.sid]['user_id'] = user_id
    
    # Notify user of successful join
    emit('joined_meeting', {
        'room_id': room_id,
        'user_id': user_id,
        'participant_count': meeting.get_participant_count(),
        'participants': [
            {'user_id': uid, 'username': pdata['username']} 
            for uid, pdata in meeting.participants.items()
        ]
    })
    
    # Notify other participants
    emit_to_room_except_sender(room_id, 'participant_joined', {
        'user_id': user_id,
        'username': username,
        'participant_count': meeting.get_participant_count()
    })

@socketio.on('leave_meeting')
def handle_leave_meeting():
    session = user_sessions.get(request.sid)
    if not session or not session['room_id']:
        return
        
    room_id = session['room_id']
    user_id = session['user_id']
    
    leave_room(room_id)
    
    if room_id in active_meetings:
        active_meetings[room_id].remove_participant(user_id)
        
        # Notify other participants
        emit_to_room(room_id, 'participant_left', {
            'user_id': user_id,
            'participant_count': active_meetings[room_id].get_participant_count()
        })
        
        # Remove room if empty
        if active_meetings[room_id].get_participant_count() == 0:
            del active_meetings[room_id]
    
    # Clear session
    session['room_id'] = None
    session['user_id'] = None

@socketio.on('video_frame')
def handle_video_frame(data):
    """Handle incoming video frame for sign language recognition"""
    session = user_sessions.get(request.sid)
    if not session or not session['room_id']:
        return
    
    room_id = session['room_id']
    user_id = session['user_id']
    frame_data = data.get('frame')
    
    if not frame_data:
        return
    
    # Process frame for sign recognition
    result = sign_processor.process_frame_fast(frame_data)
    
    if result:
        # Update meeting activity
        if room_id in active_meetings:
            meeting = active_meetings[room_id]
            meeting.update_activity()
            meeting.participants[user_id]['is_signing'] = True
            
            # Broadcast translation to all participants
            emit_to_room(room_id, 'sign_translation', {
                'user_id': user_id,
                'word': result['word'],
                'confidence': result['confidence'],
                'timestamp': result['timestamp']
            })
            
            # Reset signing status after short delay
            def reset_signing_status():
                time.sleep(2)
                if room_id in active_meetings and user_id in active_meetings[room_id].participants:
                    active_meetings[room_id].participants[user_id]['is_signing'] = False
            
            threading.Thread(target=reset_signing_status, daemon=True).start()

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk for speech recognition"""
    session = user_sessions.get(request.sid)
    if not session or not session['room_id']:
        return
    
    room_id = session['room_id']
    user_id = session['user_id']
    audio_data = data.get('audio')
    
    if not audio_data:
        return
    
    # Process audio for speech recognition
    try:
        import base64
        audio_bytes = base64.b64decode(audio_data)
        text = speech_recognizer.process_audio_chunk(audio_bytes)
        
        if text:
            # Update meeting activity
            if room_id in active_meetings:
                meeting = active_meetings[room_id]
                meeting.update_activity()
                meeting.participants[user_id]['is_speaking'] = True
                
                # Generate sign language video for the text
                sign_video_path = generate_sign_video_fast(text)
                
                if sign_video_path:
                    # Broadcast speech translation to all participants
                    emit_to_room(room_id, 'speech_translation', {
                        'user_id': user_id,
                        'text': text,
                        'video_url': f'/temp_video/{os.path.basename(sign_video_path)}',
                        'timestamp': time.time()
                    })
                
                # Reset speaking status
                def reset_speaking_status():
                    time.sleep(3)
                    if room_id in active_meetings and user_id in active_meetings[room_id].participants:
                        active_meetings[room_id].participants[user_id]['is_speaking'] = False
                
                threading.Thread(target=reset_speaking_status, daemon=True).start()
                
    except Exception as e:
        logger.error(f"Audio processing error: {e}")

@socketio.on('text_message')
def handle_text_message(data):
    """Handle text input for sign language generation"""
    session = user_sessions.get(request.sid)
    if not session or not session['room_id']:
        return
    
    room_id = session['room_id']
    user_id = session['user_id']
    text = data.get('text', '').strip()
    
    if not text:
        return
    
    try:
        # Generate sign language video
        sign_video_path = generate_sign_video_fast(text)
        
        if sign_video_path:
            # Update meeting activity
            if room_id in active_meetings:
                active_meetings[room_id].update_activity()
                
                # Broadcast to all participants
                emit_to_room(room_id, 'text_to_sign', {
                    'user_id': user_id,
                    'text': text,
                    'video_url': f'/temp_video/{os.path.basename(sign_video_path)}',
                    'timestamp': time.time()
                })
                
    except Exception as e:
        logger.error(f"Text to sign error: {e}")
        emit('error', {'message': 'Failed to generate sign language video'})

# Helper functions
def emit_to_room(room_id, event, data):
    """Emit to all participants in a room"""
    socketio.emit(event, data, room=room_id)

def emit_to_room_except_sender(room_id, event, data):
    """Emit to all participants in a room except sender"""
    socketio.emit(event, data, room=room_id, include_self=False)

def generate_sign_video_fast(text):
    """Generate sign language video with optimized speed"""
    try:
        # Use pre-generated video segments for common words
        words = text.lower().split()
        valid_words = ['hello', 'thanks', 'good', 'yes', 'no', 'help', 'sorry', 'okay']
        
        # Filter to valid words
        filtered_words = [w for w in words if w in valid_words]
        
        if not filtered_words:
            # Use default 'hello' if no valid words
            filtered_words = ['hello']
        
        # For speed, just return the first word's video
        # In production, you'd concatenate multiple videos
        word = filtered_words[0]
        
        # Create a simple video file path (would be pre-generated in production)
        video_path = os.path.join(tempfile.gettempdir(), f"sign_{word}_{uuid.uuid4()}.mp4")
        
        # Mock video generation (in production, use pre-generated videos)
        create_mock_sign_video(word, video_path)
        
        return video_path
        
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        return None

def create_mock_sign_video(word, output_path):
    """Create a mock sign video (placeholder for actual video generation)"""
    try:
        # Simple video with text overlay
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (640, 480))
        
        for i in range(20):  # 2 second video at 10fps
            # Create frame with gradient background
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 12  # Blue gradient
            frame[:, :, 1] = (20 - i) * 12  # Green gradient
            
            # Add text
            cv2.putText(frame, word.upper(), (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        
    except Exception as e:
        logger.error(f"Mock video creation error: {e}")
        raise

# REST API Routes
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'live_meeting.html')

@app.route('/temp_video/<filename>')
def serve_temp_video(filename):
    """Serve temporary video files"""
    temp_dir = tempfile.gettempdir()
    return send_from_directory(temp_dir, filename)

@app.route('/api/create_meeting', methods=['POST'])
def create_meeting():
    """Create a new meeting room"""
    room_id = str(uuid.uuid4())[:8]  # Short room ID
    active_meetings[room_id] = MeetingRoom(room_id)
    
    return jsonify({
        'room_id': room_id,
        'created_at': datetime.now().isoformat()
    })

@app.route('/api/meeting_status/<room_id>')
def meeting_status(room_id):
    """Get meeting status"""
    if room_id not in active_meetings:
        return jsonify({'error': 'Meeting not found'}), 404
    
    meeting = active_meetings[room_id]
    return jsonify({
        'room_id': room_id,
        'participant_count': meeting.get_participant_count(),
        'participants': [
            {
                'user_id': uid,
                'username': pdata['username'],
                'is_speaking': pdata['is_speaking'],
                'is_signing': pdata['is_signing']
            }
            for uid, pdata in meeting.participants.items()
        ],
        'created_at': meeting.created_at.isoformat(),
        'last_activity': meeting.last_activity.isoformat()
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'active_meetings': len(active_meetings),
        'connected_users': len(user_sessions),
        'model_available': MODEL_AVAILABLE
    })

# Cleanup background task
def cleanup_inactive_meetings():
    """Clean up inactive meetings periodically"""
    while True:
        try:
            current_time = datetime.now()
            inactive_rooms = []
            
            for room_id, meeting in active_meetings.items():
                # Remove meetings inactive for more than 1 hour
                if (current_time - meeting.last_activity).seconds > 3600:
                    inactive_rooms.append(room_id)
            
            for room_id in inactive_rooms:
                del active_meetings[room_id]
                logger.info(f"Cleaned up inactive meeting: {room_id}")
            
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            time.sleep(60)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_inactive_meetings, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    logger.info("Starting Live Meeting Sign Language Translator")
    logger.info(f"Model available: {MODEL_AVAILABLE}")
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)