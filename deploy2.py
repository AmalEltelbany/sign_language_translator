import os,uuid, time,logging,tempfile,traceback,shutil,platform,json,base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from waitress import serve
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import speech_recognition as sr
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import tensorflow as tf
from stos.sign_to_speech.model import Model
from stos.sign_to_speech.parser import Parser
import subprocess
def check_ffmpeg():
    """Check if FFmpeg is available and provide installation instructions if not"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL, 
                              check=True)
        logger.info("FFmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg not found!")
        system = platform.system().lower()
        if system == "windows":
            logger.info("To install FFmpeg on Windows:")
            logger.info("1. Download from https://ffmpeg.org/download.html")
            logger.info("2. Extract and add to PATH")
            logger.info("3. Or use: winget install FFmpeg")
        elif system == "darwin":  # macOS
            logger.info("To install FFmpeg on macOS: brew install ffmpeg")
        else:  # Linux
            logger.info("To install FFmpeg on Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return False

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using FFmpeg"""
    try:
        cmd = ['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', '-f', 'wav', output_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        logger.info(f"Successfully converted {input_path} to {output_path}")
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg first.")
        raise Exception("FFmpeg is required but not installed. Please install FFmpeg and try again.")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise Exception(f"Audio conversion failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error in audio conversion: {str(e)}")
        raise

# Flask and SocketIO setup
app = Flask(__name__, static_folder='static', template_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translation_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("translation_server")

# Meeting management
active_meetings = {}  # {room_id: {participants: {}, created_at: datetime, type: 'speaker'|'mixed'}}
meeting_history = []  # Store meeting transcripts and translations

# ---------------------------------------------
# ========== Meeting Management Classes ==========
# ---------------------------------------------
class MeetingRoom:
    def __init__(self, room_id, room_type='mixed'):
        self.room_id = room_id
        self.room_type = room_type  # 'speaker', 'non_speaker', 'mixed'
        self.participants = {}
        self.created_at = datetime.now()
        self.translation_log = []
        self.active = True

    def add_participant(self, user_id, user_type='speaker', username=None):
        """Add a participant to the meeting"""
        self.participants[user_id] = {
            'user_type': user_type,  # 'speaker' or 'non_speaker'
            'username': username or f"User-{user_id[:8]}",
            'joined_at': datetime.now(),
            'audio_enabled': True,
            'video_enabled': True
        }
        logger.info(f"Participant {user_id} ({user_type}) joined meeting {self.room_id}")

    def remove_participant(self, user_id):
        """Remove a participant from the meeting"""
        if user_id in self.participants:
            del self.participants[user_id]
            logger.info(f"Participant {user_id} left meeting {self.room_id}")

    def log_translation(self, from_user, translation_type, original_content, translated_content):
        """Log translation events"""
        self.translation_log.append({
            'timestamp': datetime.now(),
            'from_user': from_user,
            'type': translation_type,  # 'speech_to_sign', 'sign_to_speech'
            'original': original_content,
            'translated': translated_content
        })

    def get_stats(self):
        """Get meeting statistics"""
        return {
            'room_id': self.room_id,
            'participants_count': len(self.participants),
            'created_at': self.created_at.isoformat(),
            'translations_count': len(self.translation_log),
            'participants': self.participants
        }

# ---------------------------------------------
# ========== Sign to Word Translator ==========
# ---------------------------------------------
class SignToSentenceTranslator:
    def __init__(self, model_path=None, labels_path=None, sequence_length=20):
        try:
            if model_path and labels_path:
                self.model = Model(
                    stream_source=0,
                    sequence_length=sequence_length,
                    model_path=model_path,
                    labels_path=labels_path,
                    display_keypoint=False,
                    display_window=False
                )
                self.parser = Parser()
            else:
                logger.warning("Model paths not provided, using mock translator")
                self.model = None
                self.parser = None
        except Exception as e:
            logger.error(f"Failed to initialize translator: {e}")
            self.model = None
            self.parser = None

    def process_video(self, video_path):
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        try:
            if not self.model:
                # Mock response for demonstration
                return {
                    "words": ["hello", "world"],
                    "raw_sentence": "hello world",
                    "sentence": "Hello world!"
                }

            self.model.set_stream_source(video_path)

            words, all_detected_words = [], []
            frames_processed = 0
            for word, frame in self.model.start_stream():
                frames_processed += 1
                if frames_processed >= 1000:
                    break

                if word and word != "":
                    all_detected_words.append(word)
                    if word == "na":
                        words = []
                    else:
                        words.append(word)

            if not all_detected_words:
                return {"words": [], "sentence": "", "error": "No words were detected in the video"}

            display_words = [w for w in all_detected_words if w != "na"]
            raw_sentence = " ".join(display_words)

            try:
                parsed_sentence = self.parser.parse(raw_sentence) if self.parser else raw_sentence
                return {
                    "words": display_words,
                    "raw_sentence": raw_sentence,
                    "sentence": parsed_sentence
                }
            except Exception as e:
                return {
                    "words": display_words,
                    "raw_sentence": raw_sentence,
                    "sentence": raw_sentence,
                    "parsing_error": str(e)
                }

        except Exception as e:
            return {"error": str(e), "trace": traceback.format_exc()}
        finally:
            logger.info(f"Video processing done in {time.time() - start_time:.2f}s")

    def process_video_frame(self, frame_data):
        """Process a single video frame for real-time translation"""
        try:
            # Decode base64 frame data
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Save frame temporarily
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_frame_{uuid.uuid4()}.jpg")
            cv2.imwrite(temp_path, frame)
            
            # Process frame (simplified for real-time)
            if self.model and hasattr(self.model, 'process_single_frame'):
                result = self.model.process_single_frame(frame)
            else:
                # Mock result for demonstration
                result = {
                    "words": ["hello"],
                    "sentence": "Hello"
                }
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return None

# ---------------------------------------------
# ========== Text/Voice to Sign Handler ==========
# ---------------------------------------------
def create_video_directories():
    """Create necessary video directories"""
    base_dir = os.path.join("static", "dataset")
    videos_dir = os.path.join(base_dir, "videos_with_40_frames")
    output_dir = os.path.join(base_dir, "videos_with_words")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return videos_dir, output_dir

def create_sample_videos():
    """Create sample sign language videos if they don't exist"""
    videos_dir, output_dir = create_video_directories()
    
    # List of expected words
    sample_words = [
        'hello', 'thanks', 'good', 'yes', 'no', 'help', 'sorry', 'okay', 'name', 'i',
        'you', 'my', 'your', 'have', 'love', 'time', 'today', 'what', 'when', 'where'
    ]
    
    # Create a simple colored video for each word if it doesn't exist
    for word in sample_words:
        video_path = os.path.join(videos_dir, f"{word}.mp4")
        if not os.path.exists(video_path):
            logger.info(f"Creating sample video for word: {word}")
            create_sample_video(word, video_path)
    
    logger.info(f"Sample videos created in {videos_dir}")

def create_sample_video(word, output_path, duration=2, fps=10):
    """Create a simple colored video with text overlay"""
    try:
        import numpy as np
        
        # Video parameters
        width, height = 640, 480
        total_frames = duration * fps
        
        # Create a simple color based on word hash
        color_hash = hash(word) % 16777216  # 24-bit color
        color_b = color_hash & 255
        color_g = (color_hash >> 8) & 255
        color_r = (color_hash >> 16) & 255
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_num in range(total_frames):
            # Create colored frame
            frame = np.full((height, width, 3), [color_b, color_g, color_r], dtype=np.uint8)
            
            # Add text overlay
            text = word.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            
            # Get text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            # Add text with outline
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            # Add frame number indicator
            frame_text = f"Frame {frame_num + 1}/{total_frames}"
            cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Created sample video: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating sample video for {word}: {str(e)}")
        raise

def add_words_to_videos(word):
    """Add word text overlay to video"""
    videos_dir, output_dir = create_video_directories()
    base_video = os.path.join(videos_dir, f"{word}.mp4")
    output_path = os.path.join(output_dir, f"{word}.mp4")

    if not os.path.exists(base_video):
        logger.warning(f"No video found for word: {word}")
        # Try to create a sample video
        try:
            create_sample_video(word, base_video)
        except Exception as e:
            logger.error(f"Could not create sample video for {word}: {e}")
            # Return path to a default/fallback video if available
            fallback_video = os.path.join(videos_dir, "hello.mp4")
            if os.path.exists(fallback_video):
                return add_words_to_videos("hello")
            else:
                # Create a hello video as fallback
                create_sample_video("hello", fallback_video)
                return add_words_to_videos("hello")

    try:
        cap = cv2.VideoCapture(base_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {base_video}")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Ensure fps is at least 1
        
        # Use more compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add text overlay with better visibility
            text = word.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            # Add text with outline for better visibility
            cv2.putText(frame, text, (50, 50), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (50, 50), font, font_scale, (255, 255, 255), thickness)
            
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        
        logger.info(f"Created video for word '{word}' with {frame_count} frames")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating video for word '{word}': {str(e)}")
        raise

def process_text_to_sign(text):
    """Process text input and create sign language video"""
    logger.info(f"Processing text to sign: {text}")
    
    # Define valid words (expanded list)
    valid_words = {
        'age', 'book', 'call', 'car', 'day', 'egypt', 'english', 'enjoy', 'every', 'excuse',
        'football', 'forget', 'fun', 'good', 'hate', 'have', 'hello', 'help', 'holiday',
        'i', 'iam', 'love', 'meet', 'month', 'morning', 'my', 'na', 'name', 'nice', 'no',
        'not', 'number', 'okay', 'picture', 'play', 'read', 'ride', 'run', 'sorry', 'speak',
        'sport', 'take', 'thanks', 'time', 'today', 'understand', 'what', 'when', 'where',
        'year', 'yes', 'you', 'your'
    }

    # Process words and create videos
    clips = []
    processed_words = []
    
    for word in text.lower().split():
        # Clean word (remove punctuation)
        clean_word = ''.join(c for c in word if c.isalpha())
        if clean_word and clean_word in valid_words:
            try:
                logger.info(f"Processing word: {clean_word}")
                clip_path = add_words_to_videos(clean_word)
                if os.path.exists(clip_path):
                    clip = VideoFileClip(clip_path)
                    clips.append(clip)
                    processed_words.append(clean_word)
                    logger.info(f"Successfully processed word: {clean_word}")
                else:
                    logger.warning(f"Video file not created for word: {clean_word}")
            except Exception as e:
                logger.warning(f"Failed to process word: {clean_word} - {e}")

    # If no clips were created, use a default
    if not clips:
        logger.info("No words matched, creating default 'hello' video")
        try:
            default_path = add_words_to_videos("hello")
            clips = [VideoFileClip(default_path)]
            processed_words = ["hello"]
        except Exception as e:
            logger.error(f"Failed to create default video: {e}")
            raise Exception("Could not create sign language video")

    # Concatenate videos
    try:
        logger.info(f"Concatenating {len(clips)} video clips")
        final_video = concatenate_videoclips(clips)
        output_path = os.path.join(tempfile.gettempdir(), f"final_sign_{uuid.uuid4()}.mp4")
        
        # Write video with proper codec
        final_video.write_videofile(
            output_path, 
            codec="libx264", 
            audio=False,
            verbose=False,
            logger=None,
            preset='ultrafast',
            ffmpeg_params=['-crf', '23']
        )
        
        logger.info(f"Final video created: {output_path}")
        logger.info(f"Processed words: {', '.join(processed_words)}")
        
        return output_path, processed_words
        
    except Exception as e:
        logger.error(f"Video concatenation failed: {str(e)}")
        raise Exception(f"Video creation failed: {str(e)}")
    finally:
        # Clean up clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        try:
            final_video.close()
        except:
            pass

# ---------------------------------------------
# ========== SocketIO Meeting Events ==========
# ---------------------------------------------

@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to server'})

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    # Remove user from all meetings
    user_id = request.sid
    for room_id, meeting in active_meetings.items():
        if user_id in meeting.participants:
            meeting.remove_participant(user_id)
            leave_room(room_id)
            emit('user_left', {'user_id': user_id}, room=room_id)
            break

@socketio.on('create_meeting')
def on_create_meeting(data):
    """Create a new meeting room"""
    room_id = str(uuid.uuid4())[:8]
    user_type = data.get('user_type', 'speaker')  # 'speaker' or 'non_speaker'
    username = data.get('username', f'User-{request.sid[:8]}')
    
    # Create meeting room
    meeting = MeetingRoom(room_id, 'mixed')
    meeting.add_participant(request.sid, user_type, username)
    active_meetings[room_id] = meeting
    
    # Join the room
    join_room(room_id)
    
    logger.info(f"Meeting created: {room_id} by {username} ({user_type})")
    
    emit('meeting_created', {
        'room_id': room_id,
        'user_id': request.sid,
        'user_type': user_type,
        'username': username
    })

@socketio.on('join_meeting')
def on_join_meeting(data):
    """Join an existing meeting"""
    room_id = data.get('room_id')
    user_type = data.get('user_type', 'speaker')
    username = data.get('username', f'User-{request.sid[:8]}')
    
    if room_id not in active_meetings:
        emit('error', {'message': 'Meeting not found'})
        return
    
    meeting = active_meetings[room_id]
    meeting.add_participant(request.sid, user_type, username)
    
    # Join the room
    join_room(room_id)
    
    # Notify existing participants
    emit('user_joined', {
        'user_id': request.sid,
        'user_type': user_type,
        'username': username,
        'participants': meeting.participants
    }, room=room_id)
    
    # Send current participants to new user
    emit('meeting_joined', {
        'room_id': room_id,
        'participants': meeting.participants,
        'user_id': request.sid
    })
    
    logger.info(f"User {username} ({user_type}) joined meeting {room_id}")

@socketio.on('leave_meeting')
def on_leave_meeting(data):
    """Leave a meeting"""
    room_id = data.get('room_id')
    
    if room_id in active_meetings:
        meeting = active_meetings[room_id]
        meeting.remove_participant(request.sid)
        
        leave_room(room_id)
        
        # Notify other participants
        emit('user_left', {
            'user_id': request.sid,
            'participants': meeting.participants
        }, room=room_id)
        
        # Clean up empty meetings
        if len(meeting.participants) == 0:
            meeting.active = False
            meeting_history.append(meeting.get_stats())
            del active_meetings[room_id]
            logger.info(f"Meeting {room_id} ended - no participants remaining")

@socketio.on('webrtc_offer')
def on_webrtc_offer(data):
    """Handle WebRTC offer"""
    target_user = data.get('target_user')
    offer = data.get('offer')
    
    emit('webrtc_offer', {
        'from_user': request.sid,
        'offer': offer
    }, room=target_user)

@socketio.on('webrtc_answer')
def on_webrtc_answer(data):
    """Handle WebRTC answer"""
    target_user = data.get('target_user')
    answer = data.get('answer')
    
    emit('webrtc_answer', {
        'from_user': request.sid,
        'answer': answer
    }, room=target_user)

@socketio.on('webrtc_ice_candidate')
def on_webrtc_ice_candidate(data):
    """Handle WebRTC ICE candidate"""
    target_user = data.get('target_user')
    candidate = data.get('candidate')
    
    emit('webrtc_ice_candidate', {
        'from_user': request.sid,
        'candidate': candidate
    }, room=target_user)

@socketio.on('live_sign_translation')
def on_live_sign_translation(data):
    """Handle live sign language translation during meeting"""
        # Rate limiting for frame processing
    current_time = time.time()
    if not hasattr(on_live_sign_translation, 'last_process_time'):
        on_live_sign_translation.last_process_time = 0

    if current_time - on_live_sign_translation.last_process_time < 0.1:  # Process max every 500ms
        return

    on_live_sign_translation.last_process_time = current_time
    room_id = data.get('room_id')
    frame_data = data.get('frame_data')
    
    if room_id not in active_meetings:
        emit('error', {'message': 'Meeting not found'})
        return
    
    meeting = active_meetings[room_id]
    translator = app.config.get('translator')
    
    if not translator:
        emit('error', {'message': 'Translator not available'})
        return
    
    try:
        # Process the frame for sign detection
        result = translator.process_video_frame(frame_data)
        
        if result and 'sentence' in result:
            # Log the translation
            meeting.log_translation(
                request.sid, 
                'sign_to_speech', 
                result.get('words', []), 
                result.get('sentence', '')
            )
            
            # Broadcast translation to all participants
            emit('live_translation', {
                'from_user': request.sid,
                'type': 'sign_to_speech',
                'words': result.get('words', []),
                'sentence': result.get('sentence', ''),
                'timestamp': datetime.now().isoformat()
            }, room=room_id)
            
    except Exception as e:
        logger.error(f"Live sign translation error: {str(e)}")
        emit('error', {'message': 'Translation failed'})

@socketio.on('live_speech_to_sign')
def on_live_speech_to_sign(data):
    """Handle live speech to sign translation during meeting"""
    room_id = data.get('room_id')
    text = data.get('text')
    
    if room_id not in active_meetings:
        emit('error', {'message': 'Meeting not found'})
        return
    
    meeting = active_meetings[room_id]
    
    try:
        # Process text to sign video
        video_path, processed_words = process_text_to_sign(text)
        
        # Log the translation
        meeting.log_translation(
            request.sid,
            'speech_to_sign',
            text,
            processed_words
        )
        
        # Convert video to base64 for transmission
        with open(video_path, 'rb') as video_file:
            video_data = base64.b64encode(video_file.read()).decode('utf-8')
        
        # Broadcast sign video to all participants
        emit('live_translation', {
            'from_user': request.sid,
            'type': 'speech_to_sign',
            'original_text': text,
            'processed_words': processed_words,
            'video_data': video_data,
            'timestamp': datetime.now().isoformat()
        }, room=room_id)
        
        # Clean up temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)
            
    except Exception as e:
        logger.error(f"Live speech to sign error: {str(e)}")
        emit('error', {'message': 'Translation failed'})

@socketio.on('audio_data')
def on_audio_data(data):
    """Handle real-time audio for speech recognition"""
    room_id = data.get('room_id')
    audio_data = data.get('audio_data')
    
    if room_id not in active_meetings:
        return
    
    try:
        # Save audio data temporarily
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4()}.webm")
        
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Convert to WAV
        wav_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4()}.wav")
        convert_to_wav(temp_audio_path, wav_path)
        
        # Speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio)
        
        if text.strip():
            # Trigger speech to sign translation
            on_live_speech_to_sign({
                'room_id': room_id,
                'text': text
            })
        
        # Clean up
        for temp_file in [temp_audio_path, wav_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")

# ---------------------------------------------
# ========== REST API Routes ==========
# ---------------------------------------------

# Add these routes to your deploy2.py file, around line 750 where the other routes are defined

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory(app.static_folder, 'main.html')

@app.route('/meeting')
def meeting():
    """Serve the meeting page"""
    return send_from_directory(app.static_folder, 'meeting.html')

# ADD THESE NEW ROUTES:
@app.route('/voicetosign')
def voicetosign():
    """Serve the voice to sign page"""
    return send_from_directory(app.static_folder, 'voicetosign.html')

@app.route('/signtoword')
def signtoword():
    """Serve the sign to word page"""
    return send_from_directory(app.static_folder, 'signtoword.html')

@app.route('/wordtosign')
def wordtosign():
    """Serve the word to sign page"""
    return send_from_directory(app.static_folder, 'wordtosign.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)
@app.route('/convert_text_to_sign', methods=['POST'])
def convert_text_to_sign():
    """Convert text input to sign language video"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        logger.info(f"Converting text to sign: {text}")
        
        # Process text and create video
        video_path, processed_words = process_text_to_sign(text)
        
        logger.info(f"Text to sign conversion completed: {processed_words}")
        return send_file(
            video_path,
            as_attachment=True,
            download_name=f"sign_language_{int(time.time())}.mp4",
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logger.error(f"Text to sign conversion failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/convert_sign_to_text', methods=['POST'])
def convert_sign_to_text():
    """Convert sign language video to text"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400

        # Save uploaded video
        temp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}.mp4")
        video_file.save(temp_path)

        logger.info(f"Processing uploaded video: {temp_path}")
        
        # Get translator instance
        translator = app.config.get('translator')
        if not translator:
            return jsonify({"error": "Translator not available"}), 500

        # Process video
        result = translator.process_video(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Sign to text conversion failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/convert_audio_to_sign', methods=['POST'])
def convert_audio_to_sign():
    """Convert audio input to sign language video"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400

        # Save uploaded audio
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"upload_audio_{uuid.uuid4()}")
        audio_file.save(temp_audio_path)

        # Convert to WAV format
        wav_path = os.path.join(tempfile.gettempdir(), f"converted_{uuid.uuid4()}.wav")
        convert_to_wav(temp_audio_path, wav_path)

        # Speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        logger.info(f"Recognized text: {text}")

        # Convert text to sign video
        video_path, processed_words = process_text_to_sign(text)

        # Clean up audio files
        for temp_file in [temp_audio_path, wav_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return send_file(
            video_path,
            as_attachment=True,
            download_name=f"sign_from_audio_{int(time.time())}.mp4",
            mimetype='video/mp4'
        )
        
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition service error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Audio to sign conversion failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    """Get list of active meetings"""
    meetings_data = []
    for room_id, meeting in active_meetings.items():
        meetings_data.append(meeting.get_stats())
    
    return jsonify({
        "active_meetings": meetings_data,
        "total_active": len(active_meetings)
    })

@app.route('/api/meetings/<room_id>/stats', methods=['GET'])
def get_meeting_stats(room_id):
    """Get detailed stats for a specific meeting"""
    if room_id not in active_meetings:
        return jsonify({"error": "Meeting not found"}), 404
    
    meeting = active_meetings[room_id]
    return jsonify(meeting.get_stats())

@app.route('/api/meetings/<room_id>/history', methods=['GET'])
def get_meeting_history(room_id):
    """Get translation history for a meeting"""
    if room_id not in active_meetings:
        return jsonify({"error": "Meeting not found"}), 404
    
    meeting = active_meetings[room_id]
    history = []
    
    for log_entry in meeting.translation_log:
        history.append({
            'timestamp': log_entry['timestamp'].isoformat(),
            'from_user': log_entry['from_user'],
            'type': log_entry['type'],
            'original': log_entry['original'],
            'translated': log_entry['translated']
        })
    
    return jsonify({
        "room_id": room_id,
        "translation_history": history,
        "total_translations": len(history)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_meetings": len(active_meetings),
        "ffmpeg_available": check_ffmpeg()
    })

@app.route('/api/words', methods=['GET'])
def get_available_words():
    """Get list of available sign language words"""
    valid_words = [
        'age', 'book', 'call', 'car', 'day', 'egypt', 'english', 'enjoy', 'every', 'excuse',
        'football', 'forget', 'fun', 'good', 'hate', 'have', 'hello', 'help', 'holiday',
        'i', 'iam', 'love', 'meet', 'month', 'morning', 'my', 'na', 'name', 'nice', 'no',
        'not', 'number', 'okay', 'picture', 'play', 'read', 'ride', 'run', 'sorry', 'speak',
        'sport', 'take', 'thanks', 'time', 'today', 'understand', 'what', 'when', 'where',
        'year', 'yes', 'you', 'your'
    ]
    
    return jsonify({
        "available_words": sorted(valid_words),
        "total_words": len(valid_words)
    })

# ---------------------------------------------
# ========== Error Handlers ==========
# ---------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# ---------------------------------------------
# ========== Initialization and Startup ==========
# ---------------------------------------------

def initialize_translator():
    """Initialize the sign language translator"""
    try:
        # Try to initialize with actual model paths if available
        model_path = os.environ.get('SIGN_MODEL_PATH')
        labels_path = os.environ.get('SIGN_LABELS_PATH')
        
        translator = SignToSentenceTranslator(
            model_path=model_path,
            labels_path=labels_path,
            sequence_length=20
        )
        
        app.config['translator'] = translator
        logger.info("Translator initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to initialize translator with model: {e}")
        # Initialize with mock translator
        translator = SignToSentenceTranslator()
        app.config['translator'] = translator
        logger.info("Mock translator initialized")

def cleanup_temp_files():
    """Clean up temporary files on startup"""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if (filename.startswith('upload_') or 
                filename.startswith('temp_') or 
                filename.startswith('final_sign_')):
                file_path = os.path.join(temp_dir, filename)
                try:
                    os.remove(file_path)
                except:
                    pass
        logger.info("Temporary files cleaned up")
    except Exception as e:
        logger.warning(f"Could not clean up temporary files: {e}")

def main():
    """Main function to start the server"""
    try:
        # Check system requirements
        if not check_ffmpeg():
            logger.warning("FFmpeg not available - audio conversion may fail")
        
        # Clean up any leftover temporary files
        cleanup_temp_files()
        
        # Create necessary directories and sample videos
        create_sample_videos()
        
        # Initialize translator
        initialize_translator()
        
        # Get configuration
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        if debug:
            # Development server
            socketio.run(app, host=host, port=port, debug=True)
        else:
            # Production server with Waitress
            logger.info("Running in production mode with Waitress")
            serve(app, host=host, port=port, threads=4)
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()