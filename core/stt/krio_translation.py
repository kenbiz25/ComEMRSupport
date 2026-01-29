from googletrans import Translator

translator = Translator()

def krio_to_english(text: str) -> str:
    # Translate from Krio to English
    return translator.translate(text, src='kr', dest='en').text
