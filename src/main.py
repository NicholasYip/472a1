from functions import count_labels, print_graph, get_static_file
import json
import numpy as np
from pathlib import Path


file = open(get_static_file('goemotions.json'), "r")
comments = np.array(json.load(file))

emotions = count_labels(1, comments)
sentiments = count_labels(2, comments)

print_graph(emotions, 'emotions.pdf')
print_graph(sentiments, 'sentiments.pdf')

file.close()
