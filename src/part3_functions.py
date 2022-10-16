import gensim.downloader as api
import numpy as np
from nltk import tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


# import nltk
# nltk.download('punkt')


def part3(train_batch_items, test_batch_items):
    print("\nPart 3.1")
    model = api.load("word2vec-google-news-300")
    print("Pretrained word2vec-google-news-300 model loaded")

    print("\nPart 3.2")
    train_tokenized_comments = [word_tokenize(i) for i in train_batch_items.get("train_batch_comments")]
    train_unique_tokenized_comments = set([item for sublist in train_tokenized_comments for item in sublist])
    number_train_unique_tokens = len(train_unique_tokenized_comments)

    test_tokenized_comments = [word_tokenize(i) for i in test_batch_items.get("test_batch_comments")]
    test_unique_tokenized_comments = set([item for sublist in test_tokenized_comments for item in sublist])
    number_test_unique_tokens = len(test_unique_tokenized_comments)

    print("There are", number_train_unique_tokens, "different unique tokens in the training set")

    print("\nPart 3.3")
    train_embedded_comments = np.zeros(train_tokenized_comments.size)
    for i, tokenized_comment in enumerate(train_tokenized_comments):
        train_embedded_comments[i] = model.get_mean_vector(tokenized_comment)

    test_embedded_comments = np.zeros(test_tokenized_comments.size)
    for i, tokenized_comment in enumerate(test_tokenized_comments):
        test_embedded_comments[i] = model.get_mean_vector(tokenized_comment)

    print("\nPart 3.4")
    counter = 0
    for word in train_unique_tokenized_comments:
        if model.__contains__(word):
            counter = counter + 1
    print(counter, "words in the training set have an embedding found in Word2Vec, out of a total of",
          number_train_unique_tokens, ", which is", counter / number_train_unique_tokens, "%")

    counter = 0
    for word in test_unique_tokenized_comments:
        if model.__contains__(word):
            counter = counter + 1
    print(counter, "words in the test set have an embedding found in Word2Vec out of a total of ",
          number_test_unique_tokens, ", which is", counter / number_test_unique_tokens, "%")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_embedded_comments)
    clf = MLPClassifier(random_state=1, max_iter=2)

    print('\n Part 3.5 - Emotions')
    clf.fit(X, train_batch_items.get("train_batch_emotions"))

    print('\n Part 3.5 - Sentiments')
    clf.fit(X, train_batch_items.get("train_batch_sentiments"))


    search_space = {
        "activation": ["relu"],
        "hidden_layer_sizes": [(10, 10, 10), (30, 50)],
        "solver": ["adam", "sgd"]
    }
    gs = GridSearchCV(estimator=clf, param_grid=search_space)

    print("\nPart 2.3.6 - Emotions")
    gs.fit(X, train_batch_items.get("train_batch_emotions"))
    best_clf_hyperparam = gs.best_params_
    print(best_clf_hyperparam)
    # clf_improved = MLPClassifier(activation=best_clf_hyperparam["activation"],
    #                              hidden_layer_sizes=best_clf_hyperparam["hidden_layer_sizes"],
    #                              solver=best_clf_hyperparam["solver"])
    # improved_model = clf_improved.fit(X, train_batch_items.get("train_batch_emotions"))

    print("\nPart 2.3.6 - Sentiments")
    gs.fit(X, train_batch_items.get("train_batch_sentiments"))
    best_clf_hyperparam = gs.best_params_activation
    print(best_clf_hyperparam)
    # clf_improved = MLPClassifier(activation=best_clf_hyperparam[""],
    #                              hidden_layer_sizes=best_clf_hyperparam["hidden_layer_sizes"],
    #                              solver=best_clf_hyperparam["solver"])
    # improved_model = clf_improved.fit(X, train_batch_items.get("train_batch_emotions"))