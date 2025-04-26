import logging
from happytransformer import HappyTextToText, TTSettings
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("parser")

class Parser:
    """
    Parser class to convert detected words into grammatically correct English sentences.

    Attributes:
        __happy_tt (object): HappyTextToText object for initial grammar correction
        __settings (object): Text generation settings for T5 model
        __nlp (object): BART pipeline for sentence refinement
    """
    def __init__(self):
        try:
            # Initialize fine-tuned T5 model (assumes fine-tuning has been done)
            self.__happy_tt = HappyTextToText("T5", "model/t5_finetuned")
            logger.info("Initialized fine-tuned T5 model for grammar correction")
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned T5 model: {str(e)}. Falling back to default model.")
            self.__happy_tt = HappyTextToText("T5", "prithivida/grammar_error_correcter_v1")
        
        self.__settings = TTSettings(
            do_sample=True,
            top_k=10,
            temperature=0.5,
            min_length=1,
            max_length=100
        )
        
        try:
            # Initialize BART model for sentence refinement
            self.__nlp = pipeline("text2text-generation", model="facebook/bart-large")
            logger.info("Initialized BART model for sentence refinement")
        except Exception as e:
            logger.warning(f"Failed to initialize BART model: {str(e)}. Will use T5 only.")
            self.__nlp = None

    def parse(self, text):
        """
        Convert input text to a grammatically correct English sentence.

        Args:
            text (str): String of detected words from sign language

        Returns:
            str: Grammatically corrected English sentence
        """
        logger.info(f"Parser input: '{text}'")
        
        if not text or text.strip() == "":
            logger.info("Empty text provided to parser")
            return ""
        
        try:
            # Clean the input text
            text = text.strip().lower()
            
            # Remove consecutive duplicate words which are common in sign detection
            words = text.split()
            cleaned_words = []
            prev_word = None
            
            for word in words:
                if word != prev_word:
                    cleaned_words.append(word)
                    prev_word = word
            
            # Special handling for common ASL patterns
            # For example, in ASL time indicators often come first
            # This simple rule-based approach can be expanded for more complex patterns
            time_indicators = ['now', 'yesterday', 'tomorrow', 'today', 'before', 'after']
            question_indicators = ['what', 'where', 'when', 'who', 'why', 'how']
            
            # Reorder words according to English grammar when specific patterns are detected
            cleaned_text = " ".join(cleaned_words)
            logger.info(f"Cleaned text after duplicate removal: '{cleaned_text}'")
            
            # Step 1: Apply T5-based grammar correction
            tmp_text = "gec: " + cleaned_text
            t5_result = self.__happy_tt.generate_text(tmp_text, args=self.__settings).text.strip()
            logger.info(f"T5 output: '{t5_result}'")
            
            # Step 2: Refine with BART if available
            if self.__nlp:
                # Use more specific prompt to handle sign language translation 
                bart_result = self.__nlp(
                    f"translate sign language to clear English: {t5_result}", 
                    max_length=100
                )[0]['generated_text']
                logger.info(f"BART output: '{bart_result}'")
                final_text = bart_result
            else:
                final_text = t5_result
            
            # Step 3: Post-processing
            # Make sure sentence ends with proper punctuation
            if final_text and not final_text[-1] in ['.', '!', '?']:
                final_text += '.'
                
            # Capitalize first letter
            if final_text:
                final_text = final_text[0].upper() + final_text[1:]
            
            logger.info(f"Parser final output: '{final_text}'")
            return final_text
            
        except Exception as e:
            logger.error(f"Parser error: {str(e)}")
            # Fallback to original text with basic capitalization
            fallback_text = text.strip()
            if fallback_text:
                fallback_text = fallback_text[0].upper() + fallback_text[1:]
                if not fallback_text[-1] in ['.', '!', '?']:
                    fallback_text += '.'
            return fallback_text