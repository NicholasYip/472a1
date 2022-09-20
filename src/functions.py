from matplotlib import pyplot as plt
from pathlib import Path


def get_static_file(file_name):
    return Path("static/" + file_name)


def count_labels(index, arr):
    labels = {}
    for item in arr:
        key = item[index]
        if key in labels:
            labels[key] += 1
        elif key not in labels:
            labels[key] = 0
    return labels


def print_graph(labels_dict, save_name='default.pdf'):
    x_axis = list(labels_dict.keys())
    y_axis = list(labels_dict.values())
    plt.barh(x_axis, y_axis)  # printed vertically to better show the labels
    plt.savefig(get_static_file(save_name), bbox_inches="tight")
    plt.show()

