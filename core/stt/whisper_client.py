import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_path: str):
    result = model.transcribe(audio_path)
    text = result["text"]
    confidence = result.get("confidence", 0.75)  # default if not returned
    lang = result.get("language", "en")
    return text, confidence, lang
