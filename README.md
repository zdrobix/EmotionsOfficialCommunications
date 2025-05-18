# EmotionsOfficialCommunications
## About the project
#### Input
-  a video of an official press communication, and the text
#### Output
- the conveyed emotions in text, facial expression, and voice tonality

#### Why is AI needed to resolve the issue? 
- because of the large volume of official press communication in any topic (politics, sports, fashion, tech) may threaten our capability to understand the real meaning behind those talks
- seeing the lack of confidence in the politic class

---

## Training data
#### Emotion from text
- texts from tweets and labeled emotion (sadness, joy, love, anger, fear, surprise)
- 410,000 entries: 

| Emotion   | Total Count |
|-----------|-------------|
| 😔Sadness   | 121,187      |
| 😃Joy       | 141,067      |
| 💌Love      | 34,554       |
| 😡Anger     | 57,317     |
| 😨Fear      | 47,712      |
| 😲Surprise  | 14,972       |

[Link to the dataset](https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification/input)

#### Emotion from facial expression
- pictures with facial expression and labeled emotion (neutral, anger, contempt, fear, joy, surprise, sadness, disgust)

| Emotion   | Total Count |
|-----------|-------------|
| 😔Sadness   | 18      |
| 😃Joy       | 18      |
| 😡Anger     | 18     |
| 😨Fear      | 18      |
| 😲Surprise  | 18     |
| 😐Neutral  | 18     |
| 😖Contempt  | 18     |
| 🤢Disgust  | 18     |

[Link to the dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition/)

#### Emotion from audio
- audio and labeled emotion (anger, joy, sad, neutral, fear, disgust, surprise, sadness)

| Emotion    | Record Count |
|------------|--------------|
| 😡Angry      | 2167         |
| 😃Joy      | 2167         |
| 😔Sadness        | 2167         |
| 😐Neutral    | 1795         |
| 😨Fear    | 2047         |
| 🤢Disgust  | 1863         |
| 😲Surprise  | 592          |

[Link to the dataset](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions)

<br>

---

## Logistic regression for text-based emotions

Accuracy: 85%

#### Confusion Matrix
|            | Sadness | Joy   | Love  | Anger | Fear  | Surprise |
|------------|---------|------|------|------|------|----------|
| **Sadness**  | ✅ 27361 | 1728 | 123 | 725  | 413  | 63  |
| **Joy**      |  828   | ✅ 32673 | 1196 | 224  | 207  | 197  |
| **Love**     |  224   | 2039 | ✅ 6149 | 92  | 20  | 19  |
| **Anger**    | 1092   | 1436 | 82  | ✅ 11266 | 442  | 31  |
| **Fear**     |  816   | 1023 | 45  | 352  | ✅ 9189 | 454  |
| **Surprise** |  121   | 425  | 18  | 29   | 590  | ✅ 2511  |


## CNN for image-based emotions

---
---
---

>>>>>>> 48efe958b717e427351bb0c29644c52c758de10f
