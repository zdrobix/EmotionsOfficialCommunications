import pandas as pd
import joblib
import re


emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def clean_data(review):
    if isinstance(review, str):
        no_punc = re.sub(r'[^\w\s]', '', review)
        return ''.join([i for i in no_punc if not i.isdigit()])
    return ''

def load_model(filename):
    model, vectorizer = joblib.load(filename)
    print(f'Model loaded from {filename}')
    return model, vectorizer

def run_model(model, vectorizer):
    while True:
        phrase = input(f'Enter phrase (/stop):   ')
        if phrase.lower() == 'stop':
            break
        new_data = pd.DataFrame({'text': [phrase]})
        new_data['text'] = new_data['text'].apply(clean_data)

        X_new = vectorizer.transform(new_data['text'])

        predictions = model.predict(X_new)

        print("Predicted emotion: ", emotions[predictions[0]])


model, vectorizer = load_model("./models/85-66931854169266_logistic_regression_model.pkl")
run_model(model, vectorizer)
