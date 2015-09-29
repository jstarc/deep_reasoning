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
    
def prepare_snli_dataset(json_data):
    dataset = []
    for example in json_data:
       sent1 = tokenize_from_parse_tree(example['sentence1_binary_parse'])
       sent2 = tokenize_from_parse_tree(example['sentence2_binary_parse'])
       dataset.append((sent1, sent2, example['gold_label']))
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
    return glove[token] if token in glove else np.random.uniform(-0.05, 0.05, dim)
    
def load_word_vecs(token_list, glove, dim):
    return np.array([load_word_vec(x, glove, dim) for x in token_list])        
        
    
#train, dev, test = load_all_snli_datasets('E:\\Janez\\Data\\snli_1.0\\')
#glove = import_glove('E:\\Janez\\Data\\snli_vectors.txt')

#X_train, y_train = prepare_vec_dataset(train, glove)
#X_dev, y_dev = prepare_vec_dataset(dev, glove)



        


    



        
        
        

