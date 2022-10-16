import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from src.functions import convert_label_to_index, count_labels


def part2_1(comments):
    print("\nPart 2.1")
    vectorizer = CountVectorizer()
    text_comments = np.array([comment[0] for comment in comments])
    cv_fit = vectorizer.fit_transform(text_comments)
    list_features = vectorizer.get_feature_names_out()
    # list_count = cv_fit.toarray().sum(axis=0)
    # features_count = dict(zip(list_features, list_count))
    print("There are ", list_features.size, " different words in the Reddit comments")


def part2_2(comments, proportion_split):
    print("\nPart 2.2")

    list_emotions = np.array(list(count_labels(1, comments).keys()))
    list_sentiments = np.array(list(count_labels(2, comments).keys()))

    train_batch, test_batch = np.array(
        train_test_split(comments, train_size=proportion_split[0], test_size=proportion_split[1], shuffle=True),
        dtype=object)

    train_batch_comments = np.array([comment[0] for comment in train_batch])
    train_batch_emotions = np.array([comment[1] for comment in train_batch])
    train_batch_emotions_indexed = convert_label_to_index(train_batch_emotions, list_emotions)
    train_batch_sentiments = np.array([comment[2] for comment in train_batch])
    train_batch_sentiments_indexed = convert_label_to_index(train_batch_sentiments, list_sentiments)

    test_batch_comments = np.array([comment[0] for comment in test_batch])
    test_batch_emotions = np.array([comment[1] for comment in test_batch])
    test_batch_sentiments = np.array([comment[2] for comment in test_batch])

    # division by 3 is needed because each comment which is an array has 3 items in it
    print("full batch size: ", int(comments.size / 3))
    print("train batch size: ", train_batch_comments.size)
    print("test batch size:", test_batch_comments.size)

    train_batch_items = {
        "train_batch_comments": train_batch_comments,
        "train_batch_emotions": train_batch_emotions,
        "train_batch_emotions_indexed": train_batch_emotions_indexed,
        "train_batch_sentiments": train_batch_sentiments,
        "train_batch_sentiments_indexed": train_batch_sentiments_indexed
    }
    test_batch_items = {
        "test_batch_comments": test_batch_comments,
        "test_batch_emotions": test_batch_emotions,
        "test_batch_sentiments": test_batch_sentiments
    }

    return train_batch_items, test_batch_items


def part2_3_1(train_batch_items, test_batch_items):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_batch_items.get("train_batch_comments"))
    classifier = MultinomialNB()

    print('\nPart 2.3.1 - Emotions')
    classifier.fit(X, train_batch_items.get("train_batch_emotions_indexed"))

    print('\nPart 2.3.1 - Sentiments')
    classifier.fit(X, train_batch_items.get("train_batch_sentiments_indexed"))

    # test_batch = vectorizer.transform(test_batch_items.get("test_batch_comments"))
    # for comment in test_batch:
    # prediction = classifier.predict(comment)


def part2_3_2(train_batch_items, test_batch_items):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_batch_items.get("train_batch_comments"))
    dtc = tree.DecisionTreeClassifier()

    print("\nPart 2.3.2 - Emotions")
    dtc.fit(X, train_batch_items.get("train_batch_emotions"))

    print("\nPart 2.3.2 - Sentiments")
    dtc.fit(X, train_batch_items.get("train_batch_sentiments"))

    # test_batch = vectorizer.transform(test_batch_items.get("test_batch_comments"))
    # for comment in test_batch:
    # prediction = dtc.predict(comment)


def part2_3_3(train_batch_items, test_batch_items):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_batch_items.get("train_batch_comments"))
    # Cannot run the function bellow without max_iter or else it would take years to get an output.
    clf = MLPClassifier(random_state=1, max_iter=5)

    print("\nPart 2.3.3 - Emotions")
    clf.fit(X, train_batch_items.get("train_batch_emotions"))

    print("\nPart 2.3.3 - Sentiments")
    clf.fit(X, train_batch_items.get("train_batch_sentiments"))

    # test_batch = vectorizer.transform(test_batch_items.get("test_batch_comments"))
    # for comment in test_batch:
    #   prediction = clf.predict(comment)


def part2_3_4(train_batch_items, test_batch_items):
    param_grid = {"alpha": [0.5, 0, 2, 1.05, 1]}
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_batch_items.get("train_batch_comments"))
    mnb_model_grid = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid)

    print("\nPart 2.3.4 - Emotions")
    mnb_model_grid.fit(X, train_batch_items.get("train_batch_emotions"))
    best_mnb_hyperparameter = mnb_model_grid.best_params_
    classifier_improved = MultinomialNB(alpha=best_mnb_hyperparameter["alpha"])
    classifier_improved.fit(X, train_batch_items.get("train_batch_emotions_indexed"))

    print("\nPart 2.3.4 - Sentiments")
    mnb_model_grid.fit(X, train_batch_items.get("train_batch_sentiments"))
    best_mnb_hyperparameter_sentiments = mnb_model_grid.best_params_
    classifier_improved = MultinomialNB(alpha=best_mnb_hyperparameter_sentiments["alpha"])
    classifier_improved.fit(X, train_batch_items.get("train_batch_sentiments_indexed"))

    # test_batch = vectorizer.transform(test_batch_items.get("test_batch_comments"))
    # for comment in test_batch:
    #     prediction = classifier_improved.predict(comment)


def part2_3_5(train_batch_items, test_batch_items):
    search_space = {
        "criterion": ["gini", "entropy"],
        "max_depth": [100, 700],
        "min_samples_split": [2, 3, 10]
    }
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_batch_items.get("train_batch_comments"))
    dtc = tree.DecisionTreeClassifier()
    gs = GridSearchCV(estimator=dtc, param_grid=search_space)

    print("\nPart 2.3.5 - Emotions")
    gs.fit(X, train_batch_items.get("train_batch_emotions"))
    best_dtc_hyperparam = gs.best_params_
    dtc_improved = tree.DecisionTreeClassifier(criterion=best_dtc_hyperparam["criterion"],
                                               max_depth=best_dtc_hyperparam["max_depth"],
                                               min_samples_split=best_dtc_hyperparam["min_samples_split"])
    improved_model = dtc_improved.fit(X, train_batch_items.get("train_batch_emotions"))

    print("\nPart 2.3.5 - Sentiments")
    gs.fit(X, train_batch_items.get("train_batch_sentiments"))
    best_dtc_hyperparam = gs.best_params_
    dtc_improved = tree.DecisionTreeClassifier(criterion=best_dtc_hyperparam["criterion"],
                                               max_depth=best_dtc_hyperparam["max_depth"],
                                               min_samples_split=best_dtc_hyperparam["min_samples_split"])
    improved_model = dtc_improved.fit(X, train_batch_items.get("train_batch_sentiments"))

    # test_batch = vectorizer.transform(test_batch_items.get("test_batch_comments"))
    # for comment in test_batch:
    #    prediction = improved_model.predict(comment)


def part2_3_6(train_batch_items, test_batch_items):
    search_space = {
        "activation": ["logistic", "tanh", "relu", "identity"],
        "hidden_layer_sizes": [(10, 10, 10), (30, 50)],
        "solver": ["adam", "sgd"]
    }
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_batch_items.get("train_batch_comments"))
    # Cannot run the function bellow without max_iter or else it would take years to get an output.
    clf = MLPClassifier(random_state=1, max_iter=5)
    gs = GridSearchCV(estimator=clf, param_grid=search_space)

    print("\nPart 2.3.6 - Emotions")
    gs.fit(X, train_batch_items.get("train_batch_emotions"))
    best_clf_hyperparam = gs.best_params_
    clf_improved = MLPClassifier(activation=best_clf_hyperparam["activation"],
                                 hidden_layer_sizes=best_clf_hyperparam["hidden_layer_sizes"],
                                 solver=best_clf_hyperparam["solver"])
    improved_model = clf_improved.fit(X, train_batch_items.get("train_batch_emotions"))

    print("\nPart 2.3.6 - Sentiments")
    gs.fit(X, train_batch_items.get("train_batch_sentiments"))
    best_clf_hyperparam = gs.best_params_
    clf_improved = MLPClassifier(activation=best_clf_hyperparam["activation"],
                                 hidden_layer_sizes=best_clf_hyperparam["hidden_layer_sizes"],
                                 solver=best_clf_hyperparam["solver"])
    improved_model = clf_improved.fit(X, train_batch_items.get("train_batch_emotions"))

    # test_batch = vectorizer.transform(test_batch_items.get("test_batch_comments"))
    # for comment in test_batch:
    #    prediction = improved_model.predict(comment)


