import os
import uuid
import time
import logging
import tempfile
import traceback
import shutil
import platform

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from waitress import serve
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import speech_recognition as sr

# TensorFlow log suppression
os.environ['GLOG_minloglevel'] = '2'
import tensorflow as tf

# Custom translation classes
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

# Flask setup
app = Flask(__name__, static_folder='static', template_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

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

# ---------------------------------------------
# ========== Sign to Word Translator ==========
# ---------------------------------------------
class SignToSentenceTranslator:
    def __init__(self, model_path, labels_path, sequence_length=20):
        self.model = Model(
            stream_source=0,
            sequence_length=sequence_length,
            model_path=model_path,
            labels_path=labels_path,
            display_keypoint=False,
            display_window=False
        )
        self.parser = Parser()

    def process_video(self, video_path):
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        try:
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
                parsed_sentence = self.parser.parse(raw_sentence)
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
            logger=None
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
        
        return send_file(video_path, as_attachment=True, download_name="sign_language_video.mp4")

    except Exception as e:
        logger.error(f"Text to sign conversion error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500

@app.route('/convert_audio_to_sign', methods=['POST'])
def convert_audio_to_sign():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file uploaded"}), 400

        # Save uploaded audio file
        raw_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.webm")
        audio_file.save(raw_audio_path)
        logger.info(f"Saved audio file: {raw_audio_path}")

        # Convert to WAV format
        wav_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        try:
            convert_to_wav(raw_audio_path, wav_path)
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

        # Speech recognition
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
            
            text = recognizer.recognize_google(audio)
            logger.info(f"Transcribed text: {text}")
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand the audio. Please speak clearly and try again."}), 400
        except sr.RequestError as e:
            return jsonify({"error": f"Speech recognition service error: {str(e)}"}), 500

        # Use the same text processing function
        try:
            video_path, processed_words = process_text_to_sign(text)
            logger.info(f"Audio to sign conversion completed: {processed_words}")
        except Exception as e:
            logger.error(f"Video creation failed: {str(e)}")
            return jsonify({"error": f"Video creation failed: {str(e)}"}), 500

        # Clean up temporary files
        try:
            os.remove(raw_audio_path)
            os.remove(wav_path)
        except:
            pass

        return send_file(video_path, as_attachment=True, download_name="sign_language_video.mp4")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# ---------------------------------------------
# ========== Sign to Word Translation ==========
# ---------------------------------------------
@app.route('/translate', methods=['POST'])
def translate_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files['video']
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_video_{uuid.uuid4()}.mp4")
    
    try:
        video_file.save(temp_path)
        logger.info(f"Saved uploaded video to: {temp_path}")

        translator = app.config['translator']
        result = translator.process_video(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Translation error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------------------------------------
# ========== Routes for Pages ==========
# ---------------------------------------------
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'main.html')

@app.route('/voicetosign')
def voicetosign_page():
    return send_from_directory(app.static_folder, 'voicetosign.html')

@app.route('/signtoword')
def signtoword_page():
    return send_from_directory(app.static_folder, 'signtoword.html')

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "ffmpeg_available": check_ffmpeg()})

# ---------------------------------------------
# ========== Run App ==========
# ---------------------------------------------
def run_server(model_path, labels_path, host='0.0.0.0', port=5000):
    # Check FFmpeg availability
    if not check_ffmpeg():
        logger.warning("FFmpeg not found. Audio conversion will fail.")
        logger.warning("Please install FFmpeg before running the server.")
    
    # Validate model paths
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        logger.warning("Sign-to-speech translation may not work.")
    if not os.path.exists(labels_path):
        logger.warning(f"Labels file not found: {labels_path}")
        logger.warning("Sign-to-speech translation may not work.")

    # Create necessary directories
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Create sample videos
    try:
        create_sample_videos()
    except Exception as e:
        logger.error(f"Failed to create sample videos: {e}")

    # Initialize translator (if model files exist)
    if os.path.exists(model_path) and os.path.exists(labels_path):
        try:
            translator = SignToSentenceTranslator(model_path, labels_path)
            app.config['translator'] = translator
            logger.info("Translator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translator: {str(e)}")
            app.config['translator'] = None
    else:
        logger.warning("Translator not initialized due to missing model files")
        app.config['translator'] = None

    print(f"Server starting on http://{host}:{port}")
    print(f"Model path: {model_path}")
    print(f"Labels path: {labels_path}")
    print(f"Static folder: {static_dir}")
    print(f"FFmpeg available: {check_ffmpeg()}")
    
    # Run the server
    serve(app, host=host, port=port)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Combined Sign Language Translator Server')
    parser.add_argument('--model', type=str, default='model/cv_model.hdf5')
    parser.add_argument('--names', type=str, default='model/names')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    try:
        run_server(args.model, args.names, args.host, args.port)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        print(f"Error: {str(e)}")