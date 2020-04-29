# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
import nltk
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    bag = make_bag(train_set, train_labels)
    probs = log_prob(bag, smoothing_parameter)

    predicted_labels = []
    pos_unknown = np.log(smoothing_parameter/(bag['pos_count'] + (smoothing_parameter * bag['pos_unique'])))
    neg_unknown = np.log(smoothing_parameter/(bag['neg_count'] + (smoothing_parameter * bag['neg_unique'])))
    for review in dev_set:
        pos_prob = np.log(pos_prior)
        neg_prob = np.log(1 - pos_prior)
        for word in review:
            pos_prob += probs['positives'].get(word, pos_unknown)
            neg_prob += probs['negatives'].get(word, neg_unknown)

        predicted_labels.append(1) if pos_prob >= neg_prob else predicted_labels.append(0)
    
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return predicted_labels

def log_prob(bag, smoothing_parameter):
    positives, negatives = dict(), dict()

    for pos in bag['pos_words'].keys():
        log_prob_pos = np.log((bag['pos_words'][pos] + smoothing_parameter) / (bag['pos_count'] + (smoothing_parameter * bag['pos_unique'])))
        positives[pos] = log_prob_pos
    for neg in bag['neg_words'].keys():
        log_prob_neg = np.log((bag['neg_words'][neg] + smoothing_parameter) / (bag['neg_count'] + (smoothing_parameter * bag['neg_unique'])))
        negatives[neg] = log_prob_neg

    return {'positives': positives,
            'negatives': negatives}

def make_bag(train_set, train_labels):
    # creating a bag of words
    pos_words, neg_words = dict(), dict()
    pos_count, neg_count, pos_unique, neg_unique = 0, 0, 0, 0

    for i in range(len(train_set)):
        for word in train_set[i]: 
            if train_labels[i]:
                pos_count += 1
                if word in pos_words:
                    pos_words[word] += 1
                else:
                    pos_words[word] = 1
                    pos_unique += 1
            else:
                neg_count += 1
                if word in neg_words:
                    neg_words[word] += 1
                else:
                    neg_words[word] = 1
                    neg_unique += 1

    return {'pos_words': pos_words,
            'neg_words': neg_words,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_unique': pos_unique + 1,
            'neg_unique': neg_unique + 1}

