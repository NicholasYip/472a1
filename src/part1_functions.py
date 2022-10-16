import json
import numpy as np
from functions import get_static_file, count_labels, print_graph


def part1_2():
    print("\nPart 1.2")
    file = open(get_static_file('goemotions.json'), "r")
    comments = np.array(json.load(file))
    file.close()
    print("Dataset loaded")
    return comments


def part1_3(comments, print_graphs):
    print("\nPart 1.3")
    emotions = count_labels(1, comments)
    list_emotions = np.array(list(emotions.keys()))
    sentiments = count_labels(2, comments)
    list_sentiments = np.array(list(sentiments.keys()))
    if print_graphs:
        print_graph(emotions, 'Emotions', 'emotions.pdf')
        print_graph(sentiments, 'Sentiment', 'sentiments.pdf')
        print("Graphs plotted")
