# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time



def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """



    # TODO: Write your code here
    dev_results = []
    _set, train_length = set_maker(train_set)
    for review in dev_set:
        counter = dict()
        for word in review:
            counter[word] = counter.get(word, 0) + 1
        max_tf = 0
        max_word = None
        for key in counter.keys():
            a = counter[key]/len(review)
            b = np.log(train_length/(1+_set[key]))
            tf_idf = a * b
            if tf_idf > max_tf:
                max_tf = tf_idf
                max_word = key
        dev_results.append(max_word)

    # return list of words (should return a list, not numpy array or similar)
    return dev_results

def set_maker(train_set):
        idf = Counter()
        for doc in train_set:
                counter = Counter()
                for word in doc:
                        counter.update({word : 1})
                for word in counter:
                        idf.update({word : 1})
        return idf, len(train_set)
