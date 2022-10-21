from part1_functions import part1_2, part1_3
from part2_functions import part2_1, part2_2, part2_3_1, part2_3_4, part2_3_3, part2_3_2, part2_3_5, part2_3_6
from part3_functions import part3_1, part3_2, part3_3, part3_4, part3_5, part3_6

# install packages by running pip install -r requirements.txt

f = open("./static/performance.txt", "a")
model_name = "word2vec-google-news-300"

comments = part1_2()

# set to 2nd argument to True to see graphs
part1_3(comments, False)

part2_1(comments)

# set 2nd argument to train_batch/test_batch proportion split, total should be = 1
train_batch_items, test_batch_items, list_emotions, list_sentiments = part2_2(comments, [0.8, 0.2])

part2_3_1(train_batch_items, test_batch_items, list_emotions, list_sentiments, f)

part2_3_2(train_batch_items, test_batch_items, list_emotions, list_sentiments, f)

part2_3_3(train_batch_items, test_batch_items, list_emotions, list_sentiments, f)

part2_3_4(train_batch_items, test_batch_items, list_emotions, list_sentiments, f)

part2_3_5(train_batch_items, test_batch_items, list_emotions, list_sentiments, f)

part2_3_6(train_batch_items, test_batch_items, list_emotions, list_sentiments, f)

print("\nFor part 2.5, see part2_5_different_split_1.py and part2_5_different_split_2.py")

model = part3_1(model_name)

train_tokenized_items, test_tokenized_items = part3_2(train_batch_items, test_batch_items)

train_embedded_comments, test_embedded_comments = part3_3(model, train_tokenized_items, test_tokenized_items)

part3_4(model, train_tokenized_items, test_tokenized_items)

part3_5(train_embedded_comments, test_embedded_comments, train_batch_items, test_batch_items, list_emotions,
        list_sentiments, f, model_name)

part3_6(train_embedded_comments, test_embedded_comments, train_batch_items, test_batch_items, list_emotions,
        list_sentiments, f, model_name)

print("\nFor part 3.8, see part3_8_different_model_1.py and part3_8_different_model_2.py")

f.close()
