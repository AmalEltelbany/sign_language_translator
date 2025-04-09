import os
import logging
import time
import traceback

os.environ['GLOG_minloglevel'] = '2'
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from waitress import serve
from stos.sign_to_speech.model import Model
from stos.sign_to_speech.parser import Parser

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translation_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("translation_server")

class SignToSentenceTranslator:
    def __init__(self, model_path, labels_path, sequence_length=20):
        self.model = Model(
            stream_source=0,  # This will be replaced with video path
            sequence_length=sequence_length,
            model_path=model_path,
            labels_path=labels_path,
            display_keypoint=False,
            display_window=False
        )
        self.parser = Parser()

    def process_video(self, video_path):
        """
        Process the video and extract sign language words and sentences
        """
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        
        try:
            # Set stream source to the uploaded video file
            self.model.set_stream_source(video_path)
            
            # Process the video and collect words
            words = []
            all_detected_words = []
            frames_processed = 0
            max_frames = 1000  # Safety limit to prevent infinite loops
            
            logger.info("Starting frame processing")
            for word, frame in self.model.start_stream():
                frames_processed += 1
                
                if frames_processed >= max_frames:
                    logger.warning(f"Reached maximum frame count limit ({max_frames})")
                    break
                
                # Log progress periodically
                if frames_processed % 100 == 0:
                    logger.info(f"Processed {frames_processed} frames")
                
                if word and word != "":
                    logger.info(f"Detected word: '{word}'")
                    all_detected_words.append(word)
                    
                    if word == "na":  # End of sentence marker
                        # Process the current sentence
                        if words:
                            raw_sentence = " ".join(words)
                            logger.info(f"Raw sentence detected: '{raw_sentence}'")
                            words = []  # Reset for next sentence
                    else:
                        words.append(word)
            
            # Process any remaining words
            all_unique_words = list(set(all_detected_words))
            
            # Create raw sentence and parse it
            logger.info(f"All detected words: {all_detected_words}")
            logger.info(f"Unique detected words: {all_unique_words}")
            
            if not all_detected_words:
                return {"words": [], "sentence": "", "error": "No words were detected in the video"}
                
            # Filter out the "na" markers for display
            display_words = [w for w in all_detected_words if w != "na"]
            
            # Build sentence from all detected words
            raw_sentence = " ".join(display_words)
            
            try:
                parsed_sentence = self.parser.parse(raw_sentence)
                logger.info(f"Raw sentence: '{raw_sentence}'")
                logger.info(f"Parsed sentence: '{parsed_sentence}'")
                
                return {
                    "words": display_words,
                    "raw_sentence": raw_sentence,
                    "sentence": parsed_sentence
                }
            except Exception as e:
                logger.error(f"Parsing error: {str(e)}")
                return {
                    "words": display_words,
                    "raw_sentence": raw_sentence,
                    "sentence": raw_sentence,
                    "parsing_error": str(e)
                }
            
        except Exception as e:
            error_msg = f"Video processing error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"error": str(e)}
        finally:
            processing_time = time.time() - start_time
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'signtoword.html')

@app.route('/translate', methods=['POST'])
def translate_video():
    if 'video' not in request.files:
        logger.error("No video file provided in request")
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    logger.info(f"Received video: {video_file.filename}, size: {video_file.content_length} bytes")
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded file
    temp_path = os.path.join(temp_dir, "temp_video.mp4")
    video_file.save(temp_path)
    logger.info(f"Saved video to: {temp_path}")
    
    try:
        # Process the video
        translator = app.config['translator']
        logger.info("Starting video translation")
        results = translator.process_video(temp_path)
        logger.info(f"Translation results: {results}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("Deleted temporary video file")
            
        return jsonify(results)
    
    except Exception as e:
        error_msg = f"Translation error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500
def run_server(model_path, labels_path, host='0.0.0.0', port=5000):
    # Make sure model and labels files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Initialize translator
    translator = SignToSentenceTranslator(model_path, labels_path)
    app.config['translator'] = translator
    
    print(f"Starting server on {host}:{port}")
    print(f"Model path: {model_path}")
    print(f"Labels path: {labels_path}")
    
    # Run the server
    serve(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sign to Sentence Translator Web Server')
    parser.add_argument('--model', type=str, default='model/cv_model.hdf5', help='Path to sign model')
    parser.add_argument('--names', type=str, default='model/names', help='Path to names file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port for server')
    args = parser.parse_args()
    
    run_server(args.model, args.names, host=args.host, port=args.port)