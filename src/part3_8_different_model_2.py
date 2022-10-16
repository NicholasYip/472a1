from src.part1_functions import part1_2
from src.part2_functions import part2_2
from src.part3_functions import part3_3, part3_2, part3_5, part3_1, part3_6


f = open("./static/performance_wiki_gigaword.txt", "w+")
model_name = "glove-wiki-gigaword-300"

comments = part1_2()

train_batch_items, test_batch_items, list_emotions, list_sentiments = part2_2(comments, [0.8, 0.2])

model = part3_1("word2vec-google-news-300")

train_tokenized_items, test_tokenized_items = part3_2(train_batch_items, test_batch_items)

train_embedded_comments, test_embedded_comments = part3_3(model, train_tokenized_items, test_tokenized_items)

part3_5(train_embedded_comments, test_embedded_comments, train_batch_items, test_batch_items, list_emotions,
        list_sentiments, f, model_name)

part3_6(train_embedded_comments, test_embedded_comments, train_batch_items, test_batch_items, list_emotions,
        list_sentiments, f, model_name)

f.close()
