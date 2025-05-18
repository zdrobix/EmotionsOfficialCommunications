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
    model_, vectorizer_ = joblib.load(filename)
    print(f'Model loaded from {filename}')
    return model_, vectorizer_

def run_model(model_, vectorizer_):
    while True:
        phrase = input(f'Enter phrase (/stop):   ')
        if phrase.lower() == 'stop':
            break
        new_data = pd.DataFrame({'text': [phrase]})
        new_data['text'] = new_data['text'].apply(clean_data)

        X_new = vectorizer_.transform(new_data['text'])

        predictions = model_.predict(X_new)

        print("Predicted emotion: ", emotions[predictions[0]])

def count_tweets():
    df = pd.read_csv("./data/twitter_emotions.csv")
    for i, emotion in enumerate(emotions):
        count = (df['label'] == i).sum()
        print(f"{emotion}: {count}")


if __name__ == '__main__':
    count_tweets()
    model, vectorizer = load_model("./models/85-66931854169266_logistic_regression_model.pkl")
    run_model(model, vectorizer)
