import gensim.downloader as api
import numpy as np
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from functions import performance_output

# import nltk
# nltk.download('punkt')


def part3_1(pretrained_model):
    print("\nPart 3.1")
    model = api.load(pretrained_model)
    print("Pretrained", pretrained_model, "model loaded")
    return model


def part3_2(train_batch_items, test_batch_items):
    print("\nPart 3.2")
    train_tokenized_comments = [word_tokenize(i) for i in train_batch_items.get("train_batch_comments")]
    train_unique_tokenized_comments = set([item for sublist in train_tokenized_comments for item in sublist])
    number_train_unique_tokens = len(train_unique_tokenized_comments)

    test_tokenized_comments = [word_tokenize(i) for i in test_batch_items.get("test_batch_comments")]
    test_unique_tokenized_comments = set([item for sublist in test_tokenized_comments for item in sublist])
    number_test_unique_tokens = len(test_unique_tokenized_comments)

    print("There are", number_train_unique_tokens, "different unique tokens in the training set")

    train_tokenized_items = {
        "train_tokenized_comments": train_tokenized_comments,
        "train_unique_tokenized_comments": train_unique_tokenized_comments,
        "number_train_unique_tokens": number_train_unique_tokens
    }
    test_tokenized_items = {
        "test_tokenized_comments": test_tokenized_comments,
        "test_unique_tokenized_comments": test_unique_tokenized_comments,
        "number_test_unique_tokens": number_test_unique_tokens
    }
    return train_tokenized_items, test_tokenized_items


def part3_3(model, train_tokenized_items, test_tokenized_items):
    print("\nPart 3.3")
    train_embedded_comments = np.array(
        [model.get_mean_vector(tokenized_comment) for tokenized_comment in
         train_tokenized_items.get("train_tokenized_comments")])
    test_embedded_comments = np.array(
        [model.get_mean_vector(tokenized_comment) for tokenized_comment in
         test_tokenized_items.get("test_tokenized_comments")])

    return train_embedded_comments, test_embedded_comments


def part3_4(model, train_tokenized_items, test_tokenized_items):
    print("\nPart 3.4")
    counter = 0
    for word in train_tokenized_items.get("train_unique_tokenized_comments"):
        if model.__contains__(word):
            counter = counter + 1
    print(counter, "words in the training set have an embedding found in Word2Vec, out of a total of",
          train_tokenized_items.get("number_train_unique_tokens"), ", which is",
          counter / train_tokenized_items.get("number_train_unique_tokens"), "%")

    counter = 0
    for word in test_tokenized_items.get("test_unique_tokenized_comments"):
        if model.__contains__(word):
            counter = counter + 1
    print(counter, "words in the test set have an embedding found in Word2Vec out of a total of ",
          test_tokenized_items.get("number_test_unique_tokens"), ", which is",
          counter / test_tokenized_items.get("number_test_unique_tokens"), "%")


def part3_5(train_embedded_comments, test_embedded_comments, train_batch_items, test_batch_items, list_emotions,
            list_sentiments, f, model_name):
    clf = MLPClassifier(random_state=1, max_iter=5)

    print('\n Part 3.5 - Emotions')
    hyperparameter = {"activation": "relu", "solver": "adam", "hidden_layer_sizes": "100"}
    clf.fit(train_embedded_comments, train_batch_items.get("train_batch_emotions"))
    predictions_emotions = np.array([clf.predict([comment])[0] for comment in test_embedded_comments])
    confusionMatrix = confusion_matrix(test_batch_items['test_batch_emotions'], predictions_emotions,
                                       labels=list_emotions)
    performance_output(f, "Base-MLP - " + model_name, "Emotions", hyperparameter, confusionMatrix,
                       classification_report(test_batch_items['test_batch_emotions'], predictions_emotions))

    print('\n Part 3.5 - Sentiments')
    clf.fit(train_embedded_comments, train_batch_items.get("train_batch_sentiments"))
    predictions_sentiments = np.array([clf.predict([comment])[0] for comment in test_embedded_comments])
    confusionMatrix = confusion_matrix(test_batch_items['test_batch_sentiments'], predictions_sentiments,
                                       labels=list_sentiments)
    performance_output(f, "Base-MLP - " + model_name, "Sentiments", hyperparameter, confusionMatrix,
                       classification_report(test_batch_items['test_batch_sentiments'], predictions_sentiments))


def part3_6(train_embedded_comments, test_embedded_comments, train_batch_items, test_batch_items, list_emotions,
            list_sentiments, f, model_name):
    clf_params = {
        "activation": "relu",
        "hidden_layer_sizes": (30, 50),
        "solver": "adam"
    }
    clf_improved = MLPClassifier(activation=clf_params["activation"],
                                 hidden_layer_sizes=clf_params["hidden_layer_sizes"],
                                 solver=clf_params["solver"])
    print("\nPart 3.6 - Sentiments")
    clf_improved.fit(train_embedded_comments, train_batch_items.get("train_batch_sentiments"))
    predictions_sentiments = np.array([clf_improved.predict([comment])[0] for comment in test_embedded_comments])
    confusionMatrix = confusion_matrix(test_batch_items['test_batch_sentiments'], predictions_sentiments,
                                       labels=list_sentiments)
    performance_output(f, "Top-MLP - " + model_name, "Sentiments", clf_params, confusionMatrix,
                       classification_report(test_batch_items['test_batch_sentiments'], predictions_sentiments))

    print("\nPart 3.6 - Emotions")
    clf_improved.fit(train_embedded_comments, train_batch_items.get("train_batch_emotions"))
    predictions_emotions = np.array([clf_improved.predict([comment])[0] for comment in test_embedded_comments])
    confusionMatrix = confusion_matrix(test_batch_items['test_batch_emotions'], predictions_emotions,
                                       labels=list_emotions)
    performance_output(f, "Top-MLP - " + model_name, "Emotions", clf_params, confusionMatrix,
                       classification_report(test_batch_items['test_batch_emotions'], predictions_emotions))
