import numpy as np
import librosa
from librosa.feature import melspectrogram
import keras

def preprocess_audio(file_path, img_size=(128, 128)):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=img_size[0])
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = librosa.util.fix_length(S_DB, size=img_size[1], axis=1) # (128, 128, 1)
    return np.expand_dims(S_DB[..., np.newaxis], axis=0)  # (1, 128, 128, 1)


def preprocess_image(img_path):
    try:
        img = keras.preprocessing.image.load_img(
            img_path,
            color_mode='grayscale',
            target_size=(48, 48)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {str(e)}")
        return None
