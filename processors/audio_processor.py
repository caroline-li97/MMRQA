import whisper
from .text_processor import TextProcessor

class AudioProcessor:

    def __init__(self):
        self.model = whisper.load_model("base")
        self.text_processor = TextProcessor()
    
    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)  # Call Whisper API
        return result["text"]  # Return raw text

    def encode(self, audio_path):
        text = self.transcribe(audio_path)
        chunks = self.text_processor.chuck_text(text)
        return [self.text_processor.encode(chunk) for chunk in chunks]