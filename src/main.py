from functions import count_labels, print_graph, get_static_file, convert_label_to_index
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# install packages by running pip install -r requirements.txt

# Part 1
print("Part 1")
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
vectorizer = CountVectorizer()
text_comments = np.array([comment[0] for comment in comments])
cv_fit = vectorizer.fit_transform(text_comments)
# list_features = vectorizer.get_feature_names_out()
# list_count = cv_fit.toarray().sum(axis=0)
# features_count = dict(zip(list_features, list_count))
# print("There are ", list_features.size, " different words in the Reddit comments\n")


# Part 2.2
print("\nPart 2.2")
train_batch, test_batch = np.array(train_test_split(comments, train_size=0.8, test_size=0.2, shuffle=True), dtype=object)

train_batch_comments = np.array([comment[0] for comment in train_batch])
train_batch_emotions = np.array([comment[1] for comment in train_batch])
train_batch_emotions_indexed = convert_label_to_index(train_batch_emotions, list_emotions)
train_batch_sentiments = np.array([comment[2] for comment in train_batch])
train_batch_sentiments_indexed = convert_label_to_index(train_batch_sentiments, list_sentiments)

test_batch_comments = np.array([comment[0] for comment in test_batch])
test_batch_emotions = np.array([comment[1] for comment in test_batch])
test_batch_sentiments = np.array([comment[2] for comment in test_batch])

print("full batch size: ", text_comments.size)
print("train batch size: ", train_batch_comments.size)
print("test batch size:", test_batch_comments.size)

# # Part 2.3
print("\nPart 2.3")
classifier = MultinomialNB()
X = vectorizer.fit_transform(train_batch_comments)
test = vectorizer.transform(test_batch_comments)

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


# print("\nPart 2.3.2 - Emotions")

# dtc = tree.DecisionTreeClassifier()

# dtc.fit(X, train_batch_emotions)
# right_predictions = 0
# for i, comment in enumerate(test):
#   prediction = dtc.predict(comment)
#   if prediction[0] == test_batch_emotions[i]:
#         right_predictions += 1
# print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_emotions.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_emotions.size)*100, '%')  

# tree_prediction = dtc.predict(X)



# print("\nPart 2.3.2 - Sentiments")
# dtc.fit(X, train_batch_sentiments)
# right_predictions = 0
# for i, comment in enumerate(test):
#     prediction = dtc.predict(comment)
#     if prediction[0] == test_batch_sentiments[i]:
#         right_predictions += 1
# print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_sentiments.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_sentiments.size)*100, '%')


print("\nPart 2.3.3 - Emotions")
# Cannot run the function bellow without max_iter or else it would take years to get an output. 
# clf = MLPClassifier(random_state=1)
clf = MLPClassifier(random_state=1, max_iter=5)
clf.fit(X, train_batch_emotions)

right_predictions = 0
for i, comment in enumerate(test):
  prediction = clf.predict(comment)
  if prediction[0] == test_batch_emotions[i]:
        right_predictions += 1
print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_emotions.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_emotions.size)*100, '%')  

print("\nPart 2.3.3 - Sentiments")
clf.fit(X, train_batch_sentiments)
right_predictions = 0
for i, comment in enumerate(test):
    prediction = clf.predict(comment)
    if prediction[0] == test_batch_sentiments[i]:
        right_predictions += 1
print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_sentiments.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_sentiments.size)*100, '%')

print("\nPart 2.3.4 - Emotions")
param_grid = {"alpha": [0.5, 0, 2, 1.05, 1]}

mnb_model_grid_emotions = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid)
mnb_model_grid_emotions.fit(X, train_batch_emotions)
best_mnb_hyperparameter_emotions = mnb_model_grid_emotions.best_params_

classifier_improved_sentiments = MultinomialNB(alpha=best_mnb_hyperparameter_emotions["alpha"])
classifier_improved_sentiments.fit(X, train_batch_emotions_indexed)
right_predictions = 0
for i, comment in enumerate(test):
    prediction = classifier_improved_sentiments.predict(comment)[0]
    if list_emotions[int(prediction)] == test_batch_emotions[i]:
        right_predictions += 1
print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_emotions.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_emotions.size)*100, '%')

print("\nPart 2.3.4 - Sentiments")
mnb_model_grid_sentiments = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid)
mnb_model_grid_sentiments.fit(X, train_batch_sentiments)
best_mnb_hyperparameter_sentiments = mnb_model_grid_sentiments.best_params_

classifier_improved_sentiments = MultinomialNB(alpha=best_mnb_hyperparameter_sentiments["alpha"])
classifier_improved_sentiments.fit(X, train_batch_sentiments_indexed)
right_predictions = 0
for i, comment in enumerate(test):
    prediction = classifier_improved_sentiments.predict(comment)[0]
    if list_sentiments[int(prediction)] == test_batch_sentiments[i]:
        right_predictions += 1
print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_sentiments.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_sentiments.size)*100, '%')

# print("\nPart 2.3.5 - Emotions")
# search_space = {
#     "criterion" : ["gini","entropy"],
#     "max_depth" : [100,700],
#     "min_samples_split": [2,3,10]

#     # "criterion" : ["gini"],
#     # "max_depth" : [700],
#     # "min_samples_split": [2]
# }

# gs = GridSearchCV(estimator=dtc, param_grid=search_space)
# gs.fit(X, train_batch_emotions)
# best_dtc_hyperparam = gs.best_params_
# print(gs.best_params_)

# dtc_improved = tree.DecisionTreeClassifier(criterion=best_dtc_hyperparam["criterion"], max_depth = best_dtc_hyperparam["max_depth"], min_samples_split = best_dtc_hyperparam["min_samples_split"])

# improved_model = dtc_improved.fit(X,train_batch_emotions)
# right_predictions = 0
# for i, comment in enumerate(test):
#   prediction = improved_model.predict(comment)
#   if prediction[0] == test_batch_emotions[i]:
#         right_predictions += 1
# print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_emotions.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_emotions.size)*100, '%') 

# print("\nPart 2.3.5 - Sentiments")
# gs.fit(X, train_batch_sentiments)
# best_dtc_hyperparam = gs.best_params_
# print(gs.best_params_)

# dtc_improved = tree.DecisionTreeClassifier(criterion=best_dtc_hyperparam["criterion"], max_depth = best_dtc_hyperparam["max_depth"], min_samples_split = best_dtc_hyperparam["min_samples_split"])
# improved_model = dtc_improved.fit(X,train_batch_sentiments)
# right_predictions = 0
# for i, comment in enumerate(test):
#     prediction = improved_model.predict(comment)
#     if prediction[0] == test_batch_sentiments[i]:
#         right_predictions += 1
# print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_sentiments.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_sentiments.size)*100, '%')


print("\nPart 2.3.6 - Emotions")
search_space = {
    "activation" : ["sigmoid","tanh","relu","identity"],
    "hidden_layer_sizes" : [(10,10,10),(30,50)],
    "solver": ["adam","sgd"]
}

gs = GridSearchCV(estimator=clf, param_grid=search_space)
gs.fit(X, train_batch_emotions)
best_clf_hyperparam = gs.best_params_
print(gs.best_params_)

clf_improved = tree.DecisionTreeClassifier(activation=best_clf_hyperparam["activation"], hidden_layer_sizes = best_clf_hyperparam["hidden_layer_sizes"], solver = best_clf_hyperparam["solver"])
improved_model = clf_improved.fit(X,train_batch_emotions)
right_predictions = 0
for i, comment in enumerate(test):
  prediction = improved_model.predict(comment)
  if prediction[0] == test_batch_emotions[i]:
        right_predictions += 1
print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_emotions.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_emotions.size)*100, '%') 


print("\nPart 2.3.6 - Sentiments")
gs.fit(X, train_batch_sentiments)
best_clf_hyperparam = gs.best_params_
print(gs.best_params_)

clf_improved = tree.DecisionTreeClassifier(activation=best_clf_hyperparam["activation"], hidden_layer_sizes = best_clf_hyperparam["hidden_layer_sizes"], solver = best_clf_hyperparam["solver"])
improved_model = clf_improved.fit(X,train_batch_emotions)
right_predictions = 0
for i, comment in enumerate(test):
    prediction = improved_model.predict(comment)
    if prediction[0] == test_batch_sentiments[i]:
        right_predictions += 1
print('# of right predictions: ', right_predictions, 'total tests: ', test_batch_sentiments.size, 'percentage accuracy: ', float(right_predictions)/float(test_batch_sentiments.size)*100, '%')

