from functions import count_labels, print_graph, get_static_file
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# install packages by running pip install -r requirements.txt

# Part 1
# print("Part 1")
file = open(get_static_file('goemotions.json'), "r")
comments = np.array(json.load(file))
file.close()

emotions = count_labels(1, comments)
sentiments = count_labels(2, comments)
# print_graph(emotions, 'emotions.pdf')
# print_graph(sentiments, 'sentiments.pdf')

# Part 2.1
# print("\nPart 2.1")
vectorizer = CountVectorizer()
text_comments = [comment[0] for comment in comments]
cv_fit = vectorizer.fit_transform(text_comments)
list_features = vectorizer.get_feature_names_out()
# list_count = cv_fit.toarray().sum(axis=0)
# features_count = dict(zip(list_features, list_count))
# print("There are ", list_features.size, " different words in the Reddit comments\n")

# Part 2.2
# print("\nPart 2.2")
train_batch, test_batch = np.array(train_test_split(comments, train_size=0.8, test_size=0.2, shuffle=True), dtype=object)
# print("full batch size: ", comments.size)
# print("train batch size: ", train_batch.size)
# print("test batch size:", test_batch.size)


# # Part 2.3
# # print("\nPart 2.3")
# print('\nPart 2.3.1 ')


#calculate partie training batch correspondant a chaque fuckin sentiment

sentiments_list = np.array([comment[2] for comment in train_batch])
sentiments_target = np.empty(sentiments_list.size);
    
for index, value in enumerate(sentiments_list):
    if value == 'negative':
        sentiments_target[index] = 0
    elif value == 'neutral':
        sentiments_target[index] = 1
    elif value == 'positive':
        sentiments_target[index] = 2
    elif value == 'ambiguous':
        sentiments_target[index] = 3

classifierSentiments = MultinomialNB()
X = vectorizer.fit_transform(np.array([comment[0] for comment in train_batch]))
model = classifierSentiments.fit(X, sentiments_target)

test = vectorizer.transform(np.array(["And I'm sure grandmas in SC love the stereotype of the violent black man."]))
print(classifierSentiments.predict(test));

# comments_only_test = np.array([comment[0] for comment in test_batch])
# test = vectorizer.transform(comments_only_test)

# counter = 0
# for i, comments in enumerate(test):
#     res = classifierSentiments.predict(comments)
#     prediction = ""
#     if res[0] == 0.:
#         prediction = 'negative'
#     elif res[0] == 1.:
#         prediction = 'neutral'
#     elif res[0] == 2.:
#         prediction = 'positive'
#     elif res[0] == 3.:
#         prediction = 'ambiguous'
#     # print('prediction: ', prediction, '||||||  answer', test_batch[i][2], '   ||||   comments: ', test_batch[i][0])
#     if prediction == test_batch[i][2]:
#         counter = counter + 1
        
# print('counter: ', counter, 'size: ', test.size)

# print('percentage accuracy: ', float(counter)/float(test.size)*100, '%')
    

