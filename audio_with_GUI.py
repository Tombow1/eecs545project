import os

os.environ["DISPLAY"] = "localhost:0"
import tkinter as tk
import pyaudio
import wave
import threading
import whisper
import torch
import time

class VoiceRecorderGUI:
    def __init__(self, root, whisper_model="base"):
        """
        A simple GUI for recording audio from the microphone and transcribing it using OpenAI Whisper.

        Args:
            root (tk.Tk): The root Tkinter window.
            whisper_model (str): Which Whisper model to load (e.g., "tiny", "base", "small", "medium", "large").
        """
        self.root = root
        self.root.title("Voice Recorder + Whisper Transcriber")

        # Audio recording state
        self.is_recording = False
        self.frames = []
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000  # Whisper typically works best at 16kHz
        self.chunk_size = 1024
        self.output_filename = "temp_audio.wav"

        # PyAudio interface
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{whisper_model}' on device={device}...")
        self.model = whisper.load_model(whisper_model, device=device)
        print("Whisper model loaded.")

        # Create GUI elements
        self.start_button = tk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(padx=10, pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(padx=10, pady=5)

        self.transcribe_button = tk.Button(self.root, text="Transcribe", command=self.transcribe_audio, state=tk.DISABLED)
        self.transcribe_button.pack(padx=10, pady=5)

        self.transcription_label = tk.Label(self.root, text="Transcription:")
        self.transcription_label.pack()

        self.transcription_text = tk.Text(self.root, wrap=tk.WORD, height=10, width=50)
        self.transcription_text.pack(padx=10, pady=5)

    def start_recording(self):
        """
        Begin recording audio from the microphone in a separate thread.
        """
        if not self.is_recording:
            self.is_recording = True
            self.stop_button.config(state=tk.NORMAL)
            self.transcribe_button.config(state=tk.DISABLED)
            self.frames = []

            # Initialize audio stream
            self.stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print("Recording started...")
            
            # Use a thread to avoid blocking the GUI
            threading.Thread(target=self.record).start()

    def record(self):
        """
        Capture audio frames while self.is_recording is True.
        """
        while self.is_recording:
            data = self.stream.read(self.chunk_size)
            self.frames.append(data)

    def stop_recording(self):
        """
        Stop recording, close the stream, and save the audio to a file.
        """
        if self.is_recording:
            self.is_recording = False
            self.stop_button.config(state=tk.DISABLED)
            self.transcribe_button.config(state=tk.NORMAL)

            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Save recorded frames as a WAV file
            wf = wave.open(self.output_filename, "wb")
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
            wf.close()

            print(f"Recording stopped. Audio saved to {self.output_filename}.")

    def transcribe_audio(self):
        """
        Use OpenAI Whisper to transcribe the recorded audio file, and display it in the text box.
        """
        if not os.path.exists(self.output_filename):
            self.transcription_text.insert(tk.END, "No audio file found. Please record first.\n")
            return

        self.transcription_text.delete("1.0", tk.END)  # clear previous text
        self.transcription_text.insert(tk.END, "Transcribing... Please wait.\n")
        self.transcription_text.update()

        def run_transcription():
            print("Transcription started...")
            result = self.model.transcribe(self.output_filename)
            text = result["text"]
            print("Transcription finished.")
            # Update the text widget
            self.transcription_text.delete("1.0", tk.END)
            self.transcription_text.insert(tk.END, text)

        # Run transcription in a separate thread so we don't block the GUI
        threading.Thread(target=run_transcription).start()

    def on_closing(self):
        """
        Cleanup PyAudio resources when closing the GUI.
        """
        if self.stream is not None and not self.stream.is_stopped():
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VoiceRecorderGUI(root, whisper_model="base")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
