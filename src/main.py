from functions import count_labels, print_graph, get_static_file, convert_label_to_index
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn import tree

# install packages by running pip install -r requirements.txt

# Part 1
# print("Part 1")
file = open(get_static_file('goemotions.json'), "r")
comments = np.array(json.load(file))
file.close()

emotions = count_labels(1, comments)
list_emotions = np.array(list(emotions.keys()))
sentiments = count_labels(2, comments)
list_sentiments = np.array(list(sentiments.keys()))
# print_graph(emotions, 'Emotions' ,'emotions.pdf')
# print_graph(sentiments, 'Sentiment' ,'sentiments.pdf')

# Part 2.1
# print("\nPart 2.1")
# vectorizer = CountVectorizer()
text_comments = np.array([comment[0] for comment in comments])
# cv_fit = vectorizer.fit_transform(text_comments)
# list_features = vectorizer.get_feature_names_out()
# list_count = cv_fit.toarray().sum(axis=0)
# features_count = dict(zip(list_features, list_count))
# print("There are ", list_features.size, " different words in the Reddit comments\n")


# Part 2.2
# print("\nPart 2.2")
train_batch, test_batch = np.array(train_test_split(comments, train_size=0.8, test_size=0.2, shuffle=True), dtype=object)

train_batch_comments = np.array([comment[0] for comment in train_batch])
train_batch_emotions = np.array([comment[1] for comment in train_batch])
train_batch_emotions_indexed = convert_label_to_index(train_batch_emotions, list_emotions)
# train_batch_sentiments = np.array([comment[2] for comment in train_batch])
# train_batch_sentiments_indexed = convert_label_to_index(train_batch_sentiments, list_sentiments)

# test_batch_comments = np.array([comment[0] for comment in test_batch])
# test_batch_emotions = np.array([comment[1] for comment in test_batch])
# test_batch_sentiments = np.array([comment[2] for comment in test_batch])

# print("full batch size: ", text_comments.size)
# print("train batch size: ", train_batch_comments.size)
# print("test batch size:", test_batch_comments.size)

# # Part 2.3
# print("\nPart 2.3")
# classifier = MultinomialNB()
# X = vectorizer.fit_transform(train_batch_comments)
# test = vectorizer.transform(test_batch_comments)


# print('\nPart 2.3.1 - Emotions')
# model = classifier.fit(X, train_batch_emotions_indexed)
# right_predictions = 0
# for i, comment in enumerate(test):
#     prediction = classifier.predict(comment)[0]
#     if list_emotions[int(prediction)] == test_batch_emotions[i]:
#         right_predictions += 1
# print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_emotions.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_emotions.size)*100, '%')

# print('\nPart 2.3.1 - Sentiments')
# model = classifier.fit(X, train_batch_sentiments_indexed)
# right_predictions = 0
# for i, comment in enumerate(test):
#     prediction = classifier.predict(comment)[0]
#     if list_sentiments[int(prediction)] == test_batch_sentiments[i]:
#         right_predictions += 1
# print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_sentiments.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_sentiments.size)*100, '%')
    

print("\nPart 2.3.2 - Emotions")

def split_into_words(line):
    import re
    word_regex_improved = r"(\w[\w']*\w|\w)"
    word_matcher = re.compile(word_regex_improved)
    return word_matcher.findall(line)

split_comments = []
encoded_comments = []
count = 0
for comment in train_batch_comments:
    split_comments.append(split_into_words(comment))

le = preprocessing.LabelEncoder()

# ## transform each word into numbers
for comment in split_comments:
    encoded_comments.append(le.fit_transform(comment))

// why the f wont the fit work
dtc = tree.DecisionTreeClassifier()
dtc.fit(np.array(encoded_comments), train_batch_emotions)




