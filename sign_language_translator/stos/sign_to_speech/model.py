import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import time

def read_labels(labels_path):
    """
    def read_labels(labels_path)

    function to read labels from names file

    Args:
        labels_path (str): names file path

    Returns: dictionary of labels, id pairs.

    """
    with open(labels_path, 'r') as names_file:
        lines = names_file.read()
    i = 0
    names_dict = {}
    for word in lines.split('\n'):
        names_dict[word] = i
        i += 1
    return names_dict


class Model:
    """
    Model class is a class to start continuous stream from the input source and classify the motions of
    sign language to words

    Attributes:
        sequence_length (int): the length of the sequence that the model already trained on

        __model (keras_model): the model that will predict the signs

        __actions (list): list of labels

        __mp_holistic_model (object): object that control the holistic model

        __mp_drawing (object): object to control the drawing utils

        __holistic (object): the holistic model to detect the landmarks

        __stream_source (int/str): the input source i.e. (camera/video)

        __display_keypoint (bool): True if you want to display landmarks on the output image
                                False otherwise

        __display_window (bool): True if you want the class to display the output window
                              False otherwise

    """

    def __init__(self, stream_source, sequence_length, model_path, labels_path,
                 display_keypoint=False, display_window=True):
        actions_map = read_labels(labels_path)

        self.__sequence_length = sequence_length

        self.__model = tf.keras.models.load_model(model_path)

        self.__actions = list(actions_map.keys())

        self.__mp_holistic_model = mp.solutions.holistic
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__holistic = self.__mp_holistic_model.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.__stream_source = stream_source

        self.__display_keypoint = display_keypoint
        self.__display_window = display_window

    def detect_keypoints(self, image):
        """
        detect the keypoints from an input image

        Args:
            image (2d-np-array): the input image

        Returns:
            image: the input image,
            results: the keypoints results

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.__holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        """
        function to draw the landmarks on the input image

        Args:
            image: input image that you want to draw the landmarks on
            results: the landmarks that you want to draw

        Returns:
            None

        """
        self.__mp_drawing.draw_landmarks(image, results.face_landmarks, self.__mp_holistic_model.FACEMESH_CONTOURS,
                                         self.__mp_drawing.DrawingSpec(color=(10, 194, 80), thickness=1, circle_radius=1),
                                         self.__mp_drawing.DrawingSpec(color=(214, 200, 80), thickness=1, circle_radius=1))
        self.__mp_drawing.draw_landmarks(image, results.pose_landmarks, self.__mp_holistic_model.POSE_CONNECTIONS,
                                         self.__mp_drawing.DrawingSpec(color=(90, 194, 80), thickness=2, circle_radius=4),
                                         self.__mp_drawing.DrawingSpec(color=(230, 200, 80), thickness=2, circle_radius=4))
        self.__mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.__mp_holistic_model.HAND_CONNECTIONS,
                                         self.__mp_drawing.DrawingSpec(color=(20, 194, 80), thickness=2, circle_radius=4),
                                         self.__mp_drawing.DrawingSpec(color=(190, 200, 80), thickness=2, circle_radius=4))
        self.__mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.__mp_holistic_model.HAND_CONNECTIONS,
                                         self.__mp_drawing.DrawingSpec(color=(20, 194, 80), thickness=2, circle_radius=4),
                                         self.__mp_drawing.DrawingSpec(color=(190, 200, 80), thickness=2, circle_radius=4))

    def extract_keypoints(self, results):
        """
        function to extrack the keypoints from the landmarks

        Args:
            results: object that contains the keypoints

        Returns:
            np-array that contain all keypoints sequentially

        """
        # face_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in
        #                            results.face_landmarks.landmark]).flatten() \
        #     if results.face_landmarks else np.zeros(468 * 3)
        pose_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in
                                   results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 4)
        right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in
                                         results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)
        left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in
                                        results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])

    # Add this method to your Model class in model.py

    def set_stream_source(self, source):
        """
        Update the stream source (e.g., for video files)
        Checks and converts video format if needed
        
        Args:
            source (str): Path to the video file
        """
        import os
        import subprocess
        import tempfile
        
        # Check if the file exists
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        
        # Check if we need to convert the video format
        # Some formats may not work well with OpenCV
        _, ext = os.path.splitext(source)
        
        # If the format is webm, convert to mp4 which works better with OpenCV
        if ext.lower() in ['.webm', '.flv', '.mov']:
            try:
                # Create a temporary file for the converted video
                fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                os.close(fd)
                
                # Convert using ffmpeg (make sure ffmpeg is installed)
                cmd = [
                    'ffmpeg', '-i', source, 
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-y', temp_path
                ]
                
                result = subprocess.run(cmd, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True)
                
                if result.returncode == 0:
                    print(f"Successfully converted video from {ext} to mp4")
                    self.__stream_source = temp_path
                    return
                else:
                    print(f"Error converting video: {result.stderr}")
                    # Fall back to original file
            except Exception as e:
                print(f"Error during video conversion: {str(e)}")
                # Fall back to original file
        
        # Use the original file path
        self.__stream_source = source

    def start_stream(self):
        """
        Start the stream for sign language detection
        
        Returns:
            Generator yielding (word, frame) tuples
        """
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.7  # Lower threshold slightly to increase sensitivity
        last_prediction_time = time.time()
        prediction_timeout = 10.0  # Increase timeout for longer videos
        
        print(f"Opening video source: {self.__stream_source}")
        cap = cv2.VideoCapture(self.__stream_source)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.__stream_source}")
            cap.release()
            yield '', None
            return
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video details: {total_frames} frames, {fps} FPS, ~{duration:.2f} seconds")
        
        frame_count = 0
        empty_frames = 0
        max_empty_frames = 30
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                
                if not success:
                    empty_frames += 1
                    print(f"Frame read failed ({empty_frames}/{max_empty_frames})")
                    
                    if empty_frames >= max_empty_frames:
                        print("Max empty frames reached, stopping stream")
                        break
                    
                    yield '', None
                    continue
                
                frame_count += 1
                empty_frames = 0
                
                if frame_count % 30 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")
                
                # Process frame
                try:
                    image, results = self.detect_keypoints(frame)
                    
                    if self.__display_keypoint:
                        self.draw_landmarks(image, results)
                    
                    keypoints = self.extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-self.__sequence_length:]
                    
                    # Make prediction when we have enough frames
                    if len(sequence) == self.__sequence_length:
                        res = self.__model.predict(np.expand_dims(sequence, axis=0))[0]
                        confidence = res[np.argmax(res)]
                        predictions.append(np.argmax(res))
                        
                        print(f"Frame {frame_count}: Top prediction confidence: {confidence:.4f}")
                        last_prediction_time = time.time()
                    
                    display = frame if not self.__display_keypoint else image
                    
                    # Check if we have consistent predictions
                    if len(predictions) >= 10:  # Require at least 10 predictions
                        recent_preds = predictions[-10:]
                        most_common = max(set(recent_preds), key=recent_preds.count)
                        count = recent_preds.count(most_common)
                        
                        if count >= 7:  # 70% of recent predictions are the same
                            current_confidence = res[most_common] if 'res' in locals() else 0
                            
                            if current_confidence > threshold:
                                word = self.__actions[most_common]
                                print(f"Detected word: '{word}' with confidence {current_confidence:.4f}")
                                
                                if len(sentence) == 0 or word != sentence[-1]:
                                    sentence.append(word)
                                    yield word, display
                                    continue
                    
                    yield '', display
                    
                except Exception as e:
                    print(f"Frame processing error: {str(e)}")
                    yield '', frame
        
        except Exception as e:
            print(f"Stream error: {str(e)}")
        finally:
            cap.release()
            if self.__display_window:
                cv2.destroyAllWindows()