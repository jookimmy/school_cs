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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as np
import math
from collections import Counter





def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
    


    # TODO: Write your code here
    bag = make_bag(train_set, train_labels)
    probs = log_prob(bag, unigram_smoothing_parameter, bigram_smoothing_parameter)

    dev_labels = []
    pos_word_unknown = np.log(unigram_smoothing_parameter/(bag['pos_count'] + (unigram_smoothing_parameter * bag['pos_unique'])))
    pos_bigram_unknown = np.log(bigram_smoothing_parameter/(bag['pos_bi_count'] + (bigram_smoothing_parameter * bag['pos_bi_unique'])))
    neg_word_unknown = np.log(unigram_smoothing_parameter/(bag['neg_count'] + (unigram_smoothing_parameter * bag['neg_unique'])))
    neg_bigram_unknown = np.log(bigram_smoothing_parameter/(bag['neg_bi_count'] + (bigram_smoothing_parameter * bag['neg_bi_unique'])))
    
    for review in dev_set:
        pos_prob = np.log(pos_prior)
        neg_prob = np.log(1 - pos_prior)
        pos_bi_prob = np.log(pos_prior)
        neg_bi_prob = np.log(1 - pos_prior)
        for i in range(len(review) - 1):
            word = review[i]
            bigram = ' '.join([review[i], review[i+1]])
            pos_prob += probs['positives'].get(word, pos_word_unknown)
            pos_bi_prob += probs['positives'].get(bigram, pos_bigram_unknown)
            neg_prob += probs['negatives'].get(word, neg_word_unknown)
            neg_bi_prob += probs['negatives'].get(bigram, neg_bigram_unknown)
        pos_prob += probs['positives'].get(review[-1], pos_word_unknown)
        neg_prob += probs['negatives'].get(review[-1], neg_word_unknown)
        
        total_pos = ((1 - bigram_lambda) * pos_prob) + (bigram_lambda * pos_bi_prob)
        total_neg = ((1 - bigram_lambda) * neg_prob) + (bigram_lambda * neg_bi_prob)

        dev_labels.append(int(total_pos >= total_neg))


    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels

def log_prob(bag, unigram_smoothing_parameter, bigram_smoothing_parameter):
    positives, negatives = dict(), dict()

    for pos in bag['pos_words'].keys():
        positives[pos] = np.log((bag['pos_words'][pos] + unigram_smoothing_parameter) / (bag['pos_count'] + (unigram_smoothing_parameter * bag['pos_unique'])))
    for neg in bag['neg_words'].keys():
        negatives[neg] = np.log((bag['neg_words'][neg] + unigram_smoothing_parameter) / (bag['neg_count'] + (unigram_smoothing_parameter * bag['neg_unique'])))
    for pos_bigram in bag['pos_bigrams'].keys():
        positives[pos_bigram] = np.log((bag['pos_bigrams'][pos_bigram] + bigram_smoothing_parameter) / (bag['pos_bi_count'] + (bigram_smoothing_parameter * bag['pos_bi_unique'])))
    for neg_bigram in bag['neg_bigrams'].keys():
        negatives[neg_bigram] = np.log((bag['neg_bigrams'][neg_bigram] + bigram_smoothing_parameter) / (bag['neg_bi_count'] + (bigram_smoothing_parameter * bag['neg_bi_unique'])))

    return {'positives' : positives,
            'negatives' : negatives}

def make_bag(train_set, train_labels):
    
    pos_words, neg_words, pos_bigrams, neg_bigrams = dict(), dict(), dict(), dict()
    pos_count, neg_count, pos_unique, neg_unique, pos_bi_count, neg_bi_count, pos_bi_unique, neg_bi_unique = 0, 0, 0, 0, 0, 0, 0, 0


    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            word = train_set[i][j]
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

            if j < len(train_set[i]) - 1:
                bigram = ' '.join([train_set[i][j], train_set[i][j+1]])
                if train_labels[i]:
                    pos_bi_count += 1
                    if bigram in pos_bigrams:
                        pos_bigrams[bigram] += 1
                    else:
                        pos_bigrams[bigram] = 1
                        pos_bi_unique += 1
                else:
                    neg_bi_count += 1
                    if bigram in neg_bigrams:
                        neg_bigrams[bigram] += 1
                    else:
                        neg_bigrams[bigram] = 1
                        neg_bi_unique += 1

    return {'pos_words'     : pos_words,
            'pos_bigrams'   : pos_bigrams,
            'pos_count'     : pos_count,
            'pos_bi_count'  : pos_bi_count,
            'pos_unique'    : pos_unique,
            'pos_bi_unique' : pos_bi_unique,
            'neg_words'     : neg_words,
            'neg_bigrams'   : neg_bigrams,
            'neg_count'     : neg_count,
            'neg_bi_count'  : neg_bi_count,
            'neg_unique'    : neg_unique,
            'neg_bi_unique' : neg_bi_unique}





