from text_emotion import load_model, clean_data
import pandas as pd
import re
import matplotlib.pyplot as plt

model, vectorizer = load_model("./models/85-66931854169266_logistic_regression_model.pkl")

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emotions_count = dict.fromkeys(emotions, 0)


with open('./input/communicate3.txt', 'r') as file:
    text = file.read()

input_text_split = re.split(r'[.!?]', text)
input_text_split = [s.strip() for s in input_text_split if s.strip()]

new_data = pd.DataFrame({'text': input_text_split})
new_data['text'] = new_data['text'].apply(clean_data)

X_new = vectorizer.transform(new_data['text'])
predictions = model.predict(X_new)

for pred in predictions:
    emotions_count[emotions[pred]] += 1

s = sum(emotions_count.values())
for k, v in emotions_count.items():
    pct = v * 100.0 / s
    print(f"{k}:  {pct}%")
