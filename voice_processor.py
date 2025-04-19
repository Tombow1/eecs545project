# voice_processor.py

import torch
import whisper

class WhisperHandler:
    """
    A handler class for using OpenAI's Whisper to transcribe audio files
    into text. You can then pass that text to your chat/memory logic.
    """
    def __init__(self, model_name="base", device=None):
        """
        Args:
            model_name (str): The Whisper model size (e.g. "tiny", "base", "small",
                              "medium", "large"). Larger models are more accurate
                              but require more resources.
            device (str or None): The device to run Whisper on. If None, automatically
                                  picks "cuda" when available, else "cpu".
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load the requested Whisper model from the openai/whisper library
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe_audio(self, audio_path):
        """
        Transcribes the audio file at `audio_path` into text.

        Args:
            audio_path (str): Path to the audio file (e.g., .wav, .mp3, .m4a).

        Returns:
            str: The transcribed text.
        """
        # Perform the transcription
        result = self.model.transcribe(audio_path)
        return result["text"]
