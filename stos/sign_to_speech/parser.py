from happytransformer import HappyTextToText
from happytransformer import TTSettings


class Parser:
    """
    Parser class convert some words to a right english statement

    Attributes:
        __happy_tt (object): happyTextToText object to correct the grammar
        __settings (object): text generator settings object
    """
    def __init__(self):
        self.__happy_tt = HappyTextToText("T5",  "prithivida/grammar_error_correcter_v1")
        self.__settings = TTSettings(do_sample=True, top_k=10, temperature=0.5, min_length=1, max_length=100)

    # Add this to the Parser class in parser.py
    def parse(self, text):
        """
        convert the input text to right english statement

        Args:
            text (str): string of some english words

        Returns:
            string: text of the right english statement
        """
        print(f"Parser input: '{text}'")
        
        if not text or text.strip() == "":
            print("Empty text provided to parser")
            return ""
        
        try:
            tmp_text = "gec: " + text
            result = self.__happy_tt.generate_text(tmp_text, args=self.__settings)
            print(f"Parser output: '{result.text}'")
            return result.text
        except Exception as e:
            print(f"Parser error: {str(e)}")
            # Fall back to original text on error
            return text