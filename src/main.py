from functions import count_labels, print_graph, get_static_file
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Part 1
file = open(get_static_file('goemotions.json'), "r")
comments = np.array(json.load(file))
file.close()

emotions = count_labels(1, comments)
sentiments = count_labels(2, comments)

# print_graph(emotions, 'emotions.pdf')
# print_graph(sentiments, 'sentiments.pdf')

# Part 2.1
#TODO: fix memory issue
text_comments = [comment[0] for comment in comments]
vectorizer = CountVectorizer()
cv_fit = vectorizer.fit_transform(text_comments)
list_features = vectorizer.get_feature_names_out()
list_count = cv_fit.toarray().sum(axis=0)
features_count = dict(zip(list_features, list_count))

#Part 2.2
train_batch, test_batch = np.array(train_test_split(comments, train_size=0.8, test_size=0.2))
# print("full batch size: ", comments.size)
# print("train batch size: ", train_batch.size)
# print("test batch size:" , test_batch.size)