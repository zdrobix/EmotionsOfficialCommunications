from matplotlib import pyplot as plt
def plot_emotions(emotion_dict, title, path):
    emotions = list(emotion_dict.keys())
    counts = list(emotion_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(emotions, counts, color='skyblue')
    plt.title(f"Emotion Distribution - {title}")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
