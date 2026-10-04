import matplotlib.pyplot as plt

def plot_emotion_graph(name, emotion_time):

    emotions = list(emotion_time.keys())
    times = list(emotion_time.values())

    plt.figure()
    plt.bar(emotions, times)

    plt.title(f"{name}'s emotions over time")
    plt.xlabel("Emotions")
    plt.ylabel("Time (seconds)")

    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()