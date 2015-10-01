# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:37:25 2015

@author: Janez
"""
import numpy as np
import json

DELIMITER = "--"
LABEL_LIST = ['neutral','contradiction','entailment']

def import_glove(filename, filter_set = None):
    word_map = dict()    
    with open(filename, "r") as f:
        for line in f:
            head, vec = import_glove_line(line)
            if filter_set == None or head in filter_set:      
                word_map[head] = vec
    return word_map

def write_glove(filename, glove):
    with open(filename, "w") as f:
        for head in glove:
            f.write(head + " " + ' '.join(np.char.mod('%.5g',glove[head])) + "\n")
        

def import_glove_line(line):
    partition = line.partition(' ')
    return partition[0], np.fromstring(partition[2], sep = ' ') 
    

def import_snli_file(filename):
    data = []   
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
    
def prepare_snli_dataset(json_data, exclude_undecided = True):
    dataset = []
    for example in json_data:
        sent1 = tokenize_from_parse_tree(example['sentence1_binary_parse'])
        sent2 = tokenize_from_parse_tree(example['sentence2_binary_parse'])
        gold = example['gold_label']
	if not excluded_undecided or gold in LABEL_LIST:
             dataset.append((sent1, sent2, gold]))
    return dataset
    
def tokenize_from_parse_tree(parse_tree):
    result = parse_tree.lower().replace('(', ' ').replace(')', ' ').split()
    result = ['(' if el=='-lrb-' else el for el in result]
    result = [')' if el=='-rrb-' else el for el in result]
    return result

def all_tokens(dataset):
    tokens = set()
    tokens.add(DELIMITER)    
    for e in dataset:
        tokens |= set(e[0])
        tokens |= set(e[1])
    return tokens
    

def repackage_glove(input_filename, output_filename, snli_path):
    train, dev, test = load_all_snli_datasets(snli_path)
    
    tokens = all_tokens(train) | all_tokens(dev) | all_tokens(test)
    glove = import_glove(input_filename, tokens)
    print "Glove imported"
    write_glove(output_filename, glove)

def load_all_snli_datasets(snli_path):
    print "Loading training data"
    train = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_train.jsonl'))
    print "Loading dev data"
    dev = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_dev.jsonl'))
    print "Loading test data"
    test = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_test.jsonl'))
    print "Data loaded"
    return train, dev, test

#repackage_glove('E:\\Janez\\Data\\vectors.6B.50d.txt', 'E:\\Janez\\Data\\snli_vectors.txt', 'E:\\Janez\\Data\\snli_1.0\\')


def prepare_vec_dataset(dataset, glove):
    X = []   
    y = []
    for example in dataset:
        if example[2] == '-':
            continue
        concat = example[0] + ["--"] + example[1]
        X.append(load_word_vecs(concat, glove, 50))
        y.append(LABEL_LIST.index(example[2]))
    one_hot_y = np.zeros((len(y), len(LABEL_LIST)))
    one_hot_y[np.arange(len(y)), y] = 1
    return np.array(X), one_hot_y
    
def load_word_vec(token, glove, dim = 50):
    if token not in glove:
	glove[token] = np.random.uniform(-0.05, 0.05, dim)    
    return glove[token]
    
def load_word_vecs(token_list, glove, dim):
    return np.array([load_word_vec(x, glove, dim) for x in token_list])        
        
def pad_sequences(sequences, maxlen=None, dim=1, dtype='float32',
    padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
#train, dev, test = load_all_snli_datasets('data\\snli_1.0\\')
#glove = import_glove('data\\snli_vectors.txt')

#X_train, y_train = prepare_vec_dataset(train, glove)
#X_dev, y_dev = prepare_vec_dataset(dev, glove)



        


    



        
        
        

