"""
Run the following:
pip install --upgrade openai-whisper
pip install torch
conda install -c conda-forge ffmpeg
"""

from typing import Tuple
import whisper

def speech_to_text(wav_file_path) -> Tuple[str, str]:
    """
    Transcribe a WAV audio file to text using OpenAI Whisper.

    Parameters
    ----------
    wav_file_path : str
        Path to the input WAV file. The file should exist and be in WAV format

    Returns
    -------
    tuple
        A tuple (language, text_detected) where:
        - language: str
            ISO 639-1 code of the detected language (e.g., "en", "ro").
        - text_detected: str
            The transcribed text from the audio file.

    Raises
    ------
    FileNotFoundError
        If the file at `wav_file_path` does not exist.
    RuntimeError
        If transcription fails due to missing dependencies (e.g., ffmpeg) or model errors.
    """
    # The potential arguments for load_model: "tiny", "base", "small", "medium", "large"
    model = whisper.load_model("tiny")
    result = model.transcribe(wav_file_path)

    language = result["language"]
    text_detected = result["text"]

    return language, text_detected


def test_speech_to_text():
    language, text = speech_to_text("SpeechData\\bush_prudent1.wav")
    assert language == "en"
    print(text)
    language, text = speech_to_text("SpeechData\\harvard.wav")
    assert language == "en"
    print(text)
    language, text = speech_to_text("SpeechData\\bush-clinton_debate_waffle.wav")
    assert language == "en"
    print(text)

