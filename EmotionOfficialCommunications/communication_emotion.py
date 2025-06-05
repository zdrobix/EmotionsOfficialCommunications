import numpy as np

from text_emotion import load_model, clean_data
from preprocess_functions import preprocess_audio, preprocess_image
from plotting_functions import plot_emotions

import pandas as pd
import re
import keras
import os

from frames_from_video import extract_frames_from_video
from speech_to_text import speech_to_text
from video_to_audio import extract_audio_from_video

model_text, vectorizer_text = load_model("./models/85-66931854169266_logistic_regression_model.pkl")
model_facial = keras.models.load_model("./models_facial/facial_emotion_model_final.keras")
model_audio = keras.models.load_model("./models/emotion_cnn_audio_only.keras")

emotions_text = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emotions_text_count = dict.fromkeys(emotions_text, 0)

emotions_facial = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
emotions_facial_count = dict.fromkeys(emotions_facial, 0)

emotion_audio = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def print_emotion_dict(title, dict):
    print(f"Emotions from {title}: \n")
    s = sum(dict.values())
    for k, v in dict.items():
        pct = v * 100.0 / s
        print(f"{k}:  {pct:.2f}%")
    print()

def emotions_for_text(text):
    input_text_split = re.split(r'[.!?]', text)
    input_text_split = [s.strip() for s in input_text_split if s.strip()]

    new_data = pd.DataFrame({'text': input_text_split})
    new_data['text'] = new_data['text'].apply(clean_data)

    X_new = vectorizer_text.transform(new_data['text'])
    predictions = model_text.predict(X_new)

    for pred in predictions:
        emotions_text_count[emotions_text[pred]] += 1

def emotions_for_facial_expression(facial_path):
    for img_name in os.listdir(facial_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(facial_path, img_name)
            if img_path is None:
                continue
            img_array = preprocess_image(img_path)
            prediction = np.argmax(model_facial.predict(img_array))
            emotions_facial_count[emotions_facial[prediction]] += 1

def emotions_for_audio(audio_path):
    audio = preprocess_audio(audio_path)
    prediction = np.argmax(model_audio.predict(audio))
    print(f"Emotion from audio: {emotion_audio[prediction]}")


name = "press_conference_4"
filename = f"{name}.mp4"

video_file_path = f"./input/video/{filename}"
output_frames_path = f"./input/frames_from_videos"
output_audio_path = f"./input/audio/{name}.wav"

extract_frames_from_video(video_file_path, output_frames_path, seconds=2)
extract_audio_from_video(video_path=video_file_path, output_path=output_audio_path)
language, text = speech_to_text(output_audio_path)

emotions_for_text(text)
print_emotion_dict("text", emotions_text_count)
emotions_for_facial_expression(output_frames_path + f"/{name}")
print_emotion_dict("facial expressions", emotions_facial_count)
emotions_for_audio(output_audio_path)


plot_emotions(emotions_facial_count, "Facial", f"./plots/facial_emotions_{name}.png")
plot_emotions(emotions_text_count, "Text", f"./plots/text_emotions_{name}.png")
