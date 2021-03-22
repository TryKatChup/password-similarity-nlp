# No word2keypress version
import itertools
import sys
import csv
import time
import compress_fasttext
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
from nltk import edit_distance


def leet_code(string: str):
    for char in string:
        if char == 'a':
            string = string.replace('a', '4')
        elif char == 'b':
            string = string.replace('b', '8')
        elif char == 'e':
            string = string.replace('e', '3')
        elif char == 'l':
            string = string.replace('l', '1')
        elif char == 'o':
            string = string.replace('o', '0')
        elif char == 's':
            string = string.replace('s', '5')
        elif char == 't':
            string = string.replace('t', '7')
        else:
            pass
    return string


# HEURISTICS
def heuristics(pwd1: str, pwd2: str):
    if leet_code(pwd1) == leet_code(pwd2):
        return True
    if pwd1.lower() == pwd2.lower():
        return True
    if edit_distance(pwd1, pwd2, transpositions=True) < 5:
        return True
    return False


def main():
    # Usage:
    # python3 PRGraph.py <filename>
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 PRGraph.py <filename> <compressed_model>")
    filename = sys.argv[1]
    compressed_model_file = sys.argv[2]
    small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(compressed_model_file)
    pos_neg_count = OrderedDict()
    prec_dict = OrderedDict()
    rec_dict = OrderedDict()

    for th in np.arange(0.0, 1.1, 0.1):
        pos_neg_count_th = {'TP': 0, 'FP': 0, 'FN': 0}
        pos_neg_count[th] = pos_neg_count_th

    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=':')
        start_time = time.time()
        for i, (user, pass_keyseq_list) in enumerate(csv_reader):
            if i % 10000 == 0:
                end_time = time.time()
                print("Processed {} lines in {} seconds.".format(i, end_time - start_time))
                start_time = end_time
            user_pass_list = eval(pass_keyseq_list)
            for pwd1, pwd2 in itertools.combinations(user_pass_list, 2):
                # Find similarity percentage using the model
                sim_score = small_model.similarity(pwd1, pwd2)
                ground_truth = heuristics(pwd1, pwd2)

                for th, pos_neg_count_th in pos_neg_count.items():
                    bin_sim_score = sim_score > th
                    if bin_sim_score and ground_truth:
                        pos_neg_count_th['TP'] += 1
                    elif bin_sim_score and not ground_truth:
                        pos_neg_count_th['FP'] += 1
                    elif not bin_sim_score and ground_truth:
                        pos_neg_count_th['FN'] += 1

    for th, pos_neg_count_th in pos_neg_count.items():
        th_prec = pos_neg_count_th['TP'] / (pos_neg_count_th['TP'] + pos_neg_count_th['FP'])
        th_rec = pos_neg_count_th['TP'] / (pos_neg_count_th['TP'] + pos_neg_count_th['FN'])
        prec_dict[th] = th_prec
        rec_dict[th] = th_rec

    x, y_p = zip(*prec_dict.items())
    _, y_r = zip(*rec_dict.items())

    plt.plot(x, y_p, label='Precision')
    plt.plot(x, y_r, label='Recall')
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.show()


if __name__ == '__main__':
    main()

