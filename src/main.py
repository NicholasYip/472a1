from part1_functions import part1_2, part1_3
from part2_functions import part2_1, part2_2, part2_3_1, part2_3_4, part2_3_3, part2_3_2, part2_3_5, part2_3_6

# install packages by running pip install -r requirements.txt

comments = part1_2()

# set to 2nd argument to True to see graphs
part1_3(comments, False)

part2_1(comments)

# set 2nd argument to train_batch/test_batch proportion split, total should be = 1
train_batch_items, test_batch_items = part2_2(comments, [0.8, 0.2])

# part2_3_1(train_batch_items, test_batch_items)
#
# part2_3_2(train_batch_items, test_batch_items)
#
# part2_3_3(train_batch_items, test_batch_items)
#
# part2_3_4(train_batch_items, test_batch_items)
#
# part2_3_5(train_batch_items, test_batch_items)

part2_3_6(train_batch_items, test_batch_items)

print("For part 2.5, see different_split1.py and different_split2.py")

