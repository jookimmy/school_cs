"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np


def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []
    
    counter, gt_tag = train_baseline(train)
    for i in range(len(test)):
        predicts.append(list())
        for word in test[i]:
            if word in counter:
                predicts[i].append((word, counter[word]['max_tag'][0]))
            else:
                predicts[i].append((word, gt_tag[0]))

    return predicts

def train_baseline(train):
    # making nested counter dictionary for word/tag
    counter = dict()
    tags = dict()
    for sentence in train:
        for word, tag in sentence:
            if word in counter:
                counter[word][tag] = counter[word].get(tag, 0) + 1
                # caching highest count of tag and the tag itself for the word
                if counter[word][tag] >= counter[word]['max_tag'][1]:
                    counter[word]['max_tag'] = [tag, counter[word][tag]]
            else:
                counter[word] = dict()
                counter[word][tag] = 1
                counter[word]['max_tag'] = [tag, 1]
            tags[tag] = tags.get(tag, 0) + 1

    gt_tag = max(tags, key = lambda x : tags[x])
    return counter, (gt_tag, tags[gt_tag])


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    
    counters = counter_init(train)
    probs = make_prob(counters, len(train), .00001)
    for sentence in test:
        predict = []
        optimal = make_trellis(probs, counters, sentence, len(train), .00001)
        for pointer in optimal[1]:
            predict.append((sentence[pointer[0]], pointer[1]))
        predicts.append(predict)

    return predicts

def make_trellis(probs, counters, sentence, count_start, K):
    trellis = []
    backpointer = dict()

    trellis.append(dict())
    for j in counters['tags_list']:
        emission = probs['emission'].get((sentence[0], j), probs['emission']['smooth'][j])
        trellis[0][j] = probs['initial'].get(j, probs['initial']['smooth']) + emission
        backpointer[(0, j)] = None
    for i in range(1, len(sentence)):
        trellis.append(dict())
        for tag in counters['tags_list']:
            prev_tag = None
            max_prob = float('-inf')
            for prev in counters['tags_list']:
                emission = probs['emission'].get((sentence[i], tag), probs['emission']['smooth'][tag])
                transition = probs['transition'].get((prev, tag), probs['transition']['smooth'][prev])
                prob = trellis[i-1][prev] + emission + transition
                if prob > max_prob:
                    prev_tag = prev
                    max_prob = prob
            trellis[i][tag] = max_prob
            backpointer[(i, tag)] = (i-1, prev_tag)
    
    best_prob = max(trellis[-1].values())
    pointer = (len(sentence) - 1, max(trellis[-1], key = lambda tag : trellis[-1][tag]))
    best_path = []
    while pointer != None:
        best_path.insert(0, pointer)
        pointer = backpointer[pointer]
        
    
    return (best_prob, best_path)


def make_prob(counters, count_start, K):
    ip, tp, ep, hp = dict(), dict(), dict(), dict()

    tp['smooth'] = dict()
    ep['smooth'] = dict()
    
    hapax_words = [key for key in counters['words'].keys() if counters['words'][key] == 1]

    for word in hapax_words:
        for tag in counters['tags_list']:
            value = hp.get(tag, 0) + counters['emission'].get((word, tag), 0)
            if value > 0:
                hp[tag] = hp.get(tag, 0) + counters['emission'].get((word, tag), 0)

    for key in counters['initial'].keys():
        ip[key] = np.log((counters['initial'][key] + K)/(count_start + (K * counters['num_tags'])))
    ip['smooth'] = np.log(K/(count_start + (K * counters['num_tags'])))
    
    for key in counters['transition'].keys():
        tp[key] = np.log((counters['transition'][key] + K)/(counters['tags'][key[0]] + (K * counters['num_tags'])))
        if key[1] not in tp['smooth']:
            tp['smooth'][key[1]] = np.log(K/(counters['tags'][key[0]] + (K * counters['num_tags'])))
    
    for key in counters['emission'].keys():
        if len(hapax_words) > 0:
            K = .00001 * hp.get(key[1], 1)/len(hapax_words)
        ep[key] = np.log((counters['emission'][key] + K)/(counters['tags'][key[1]] + (K * (counters['vocab_size'] + 1))))
        if key[1] not in ep['smooth']:
            ep['smooth'][key[1]] = np.log(K/(counters['tags'][key[1]] + (K * (counters['vocab_size'] + 1))))
    
    return { 'initial' : ip,
             'transition' : tp,
             'emission' : ep }

def counter_init(train):
    tag_counter, init_counter, em_counter, tran_counter = dict(), dict(), dict(), dict()
    word_counter = dict()
    num_tags = 0
    unique, tags = set(), set()
    for sentence in train:
        # making init_counter
        init_counter[sentence[0][1]] = init_counter.get(sentence[0][1], 0) + 1
        # making emission and transitional counters in same loop
        for i in range(len(sentence)):
            # transition counter
            if i > 0:
                prev_tag = sentence[i-1][1]
                tran = (prev_tag, sentence[i][1])
                tran_counter[tran] = tran_counter.get(tran, 0) + 1
            # emissions counter
            em_counter[sentence[i]] = em_counter.get(sentence[i], 0) + 1
            # general tag counter
            tag_counter[sentence[i][1]] = tag_counter.get(sentence[i][1], 0) + 1
            # num_tags increment
            num_tags += 1
            unique.add(sentence[i])
            tags.add(sentence[i][1])
            word_counter[sentence[i][0]] = word_counter.get(sentence[i][0], 0) + 1

    return { 'tags' : tag_counter,
             'initial' : init_counter,
             'emission' : em_counter,
             'transition' : tran_counter,
             'words' : word_counter,
             'num_tags' : num_tags,
             'tags_list' : list(tags),
             'vocab_size' : len(unique) }


def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    
    counters = counter_init(train)
    probs = make_prob(counters, len(train), .00001)

    for sentence in test:
        predict = []
        optimal = make_trellis(probs, counters, sentence, len(train), .00001)
        for pointer in optimal[1]:
            predict.append((sentence[pointer[0]], pointer[1]))
        predicts.append(predict)

    return predicts
