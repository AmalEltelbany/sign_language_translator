import os
import threading
from stos.sign_to_speech import model_prepare
from stos.sign_to_speech.model import Model
from stos.sign_to_speech.speak import Speak
from stos.sign_to_speech.parser import Parser


class SignToSpeech:
    """
    SignToSpeech class that build the pipeline of converting the sings to speech

    Attributes:
        __model (Model): model object that control the sign prediction

        __sentence_queue (list): list of detected sentences as a queue

        __listener_thread (Thread): sentence listener thread

        __speak (Speak): speak object to speak the sentences

        __parser (Parser): parser object to construct the right sentence

        __display_window (bool, optional):
            True if you want the class to display the output window
            False otherwise

    """

    def __init__(self, source, sequence_length, model_path, names_path, display_keypoint=False, display_window=True):
        model_exist = os.path.exists(model_path)
        if not model_exist:
            print('Downloading the __model.')
            model_url = 'https://drive.google.com/u/0/uc?id=1LkQWfCo4T9uAZAykKvkVs8bKub6b96LC&export=download'
            model_prepare.download_file(model_url, model_path)
            print('Downloading names file.')
            names_url = 'https://drive.google.com/u/0/uc?id=1VmT3F9X9E_kavPKheSk4q5QZjyS9bgNn&export=download'
            model_prepare.download_file(names_url, names_path)
        self.__model = Model(source, sequence_length, model_path, names_path, display_keypoint, display_window)
        self.__sentence_queue = []
        self.__listener_thread = threading.Thread(target=self.sentence_listener)
        self.__speak = Speak()
        self.__parser = Parser()
        self.__display_window = display_window

    def sentence_listener(self):
        """
        function to listen to the __sentence_queue attribute if there is a sentence it will process it.

        Returns:
            None

        """
        while len(self.__sentence_queue) > 0:
            sentence = self.__parser.parse(self.__sentence_queue[0])
            print('sentence:', self.__sentence_queue[0])
            print('parsed:', sentence)
            self.__speak.speak(sentence)
            del self.__sentence_queue[0]

    def start_pipeline(self):
        """
        this function start the whole pipeline to convert the sign language to spoken language.

        Returns:
                word (string): the predicted word.
                frame (2d-np_array): the frame that return from the stream.

        """
        words = []
        last_word = ""
        consecutive_same_word = 0
        base_confidence_threshold = 0.85  # Base confidence threshold
        min_frames_for_word = 3  # Minimum frames to confirm a word
        word_confidence_count = {}
        word_confidence_scores = {}
        dynamic_threshold = base_confidence_threshold
        
        for word, frame in self.__model.start_stream():
            if word != "" and hasattr(frame, 'confidence'):
                # Update dynamic threshold based on recent detections
                dynamic_threshold = max(base_confidence_threshold - (len(words) * 0.02), 0.75)
                
                if frame.confidence >= dynamic_threshold:
                    print(f"Detected word: {word} with confidence {frame.confidence:.4f} (threshold: {dynamic_threshold:.4f})")
                    
                    # Track confidence scores for better word validation
                    if word not in word_confidence_scores:
                        word_confidence_scores[word] = [frame.confidence]
                    else:
                        word_confidence_scores[word].append(frame.confidence)
                        # Keep only recent confidence scores
                        word_confidence_scores[word] = word_confidence_scores[word][-5:]
                    
                    # Count high-confidence detections for each word
                    if word not in word_confidence_count:
                        word_confidence_count[word] = 1
                    else:
                        word_confidence_count[word] += 1
                
                # Calculate average confidence for the word
                avg_confidence = sum(word_confidence_scores.get(word, [0])) / len(word_confidence_scores.get(word, [1]))
                
                # Process word if it meets both frequency and confidence criteria
                if word_confidence_count[word] >= min_frames_for_word and avg_confidence >= dynamic_threshold:
                    # Handle repeated words with improved logic
                    if word == last_word:
                        consecutive_same_word += 1
                        if consecutive_same_word > 2:  # Skip excessive repetitions
                            continue
                    else:
                        consecutive_same_word = 0
                        last_word = word
                        # Only reset counts for the current word to maintain context
                        word_confidence_count[word] = 0
                        word_confidence_scores[word] = []
                    
                    if word == 'na':
                        if words:  # Process sentence
                            sentence = ' '.join(words)
                            print(f"Forming sentence: {sentence}")
                            self.__sentence_queue.append(sentence)
                            words = []
                            if not self.__listener_thread.is_alive():
                                del self.__listener_thread
                                self.__listener_thread = threading.Thread(target=self.sentence_listener)
                                self.__listener_thread.start()
                    else:
                        # Add word if it passes all filters
                        if len(word) > 1:  # Skip single-character predictions
                            words.append(word)
                            print(f"Current words buffer: {words}")
            else:
                # Reset confidence count if we get a low confidence frame
                word_confidence_count = {}
            yield word, frame
