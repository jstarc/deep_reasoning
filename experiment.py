# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:43:55 2015

@author: Janez
"""

import sys
sys.path.append('../keras')

import load_data
import models
import misc

import paraphrase
import numpy as np

import itertools
import os

 

if __name__ == "__main__":
    train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
    glove = load_data.import_glove('data/snli_vectors_300.txt')


def grid_experiments(train, dev, glove, embed_size = 300, hidden_size = 100):
    lr_vec = [0.001, 0.0003, 0.0001]
    dropout_vec = [0.0, 0.1, 0.2]
    reg_vec = [0.0, 0.001, 0.0003, 0.0001]

    for params in itertools.product(lr_vec, dropout_vec, reg_vec):
	filename = 'lr' + str(params[0]).replace('.','') + '_drop' + str(params[1]).replace('.','') + '_reg' + str(params[2]).replace('.','')
	print 'Model', filename
	model = models.init_model(embed_size, hidden_size, params[0], params[1], params[2])
	models.train_model(train, dev, glove, model, 'models/' + filename)    
	

def test_model2(model, dev, glove):
    from misc import predict_example
    tp = 0
    for ex in dev:
	probs = predict_example(" ".join(ex[0]), " ".join(ex[1]), model, glove)
	label = load_data.LABEL_LIST[np.argmax(probs)]
	if label == ex[2]:
	   tp +=1
    return tp / float(len(dev))
   

def test_all_models(dev, test, glove, folder = 'models/'):
    files = os.listdir(folder)
    extless = set([file.split('.')[0] for file in files if os.path.isfile(file)]) - set([''])
    epoch_less = set([file.split('~')[0] for file in extless])
    for model_short in epoch_less:
	if model_short in extless:
	    modelname = model_short
	else:
            same_exper = [m for m in extless if m.startswith(model_short)]
	    epoch_max = max([int(file.split('~')[1]) for file in same_exper]) 
	    modelname = model_short + '~' + str(epoch_max)
	
	print modelname
	model = models.load_model(folder + modelname)
	dev_acc = models.test_model(model, dev, glove)
        test_acc = models.test_model(model, test, glove)
	print "Dev:", '{0:.2f}'.format(dev_acc * 100), "Test_acc:", '{0:.2f}'.format(test_acc * 100)
	print 
     
def accuracy_for_subset(y_pred, y_gold, subset):
    pred = y_pred[subset]
    gold =  y_gold[subset]
    return np.sum(np.argmax(pred, axis=1) == np.argmax(gold, axis=1)) / float(len(gold))

def augmented_dataset(glove, dataset, ppdb):
    new_examples = []
    for ex in dataset:
	new_examples += augment_example(glove, ex, ppdb)
    return new_examples

def augment_example(glove, example, ppdb):
    new_examples = []
    for word in set(example[0] + example[1]):
        if word in ppdb:
	      for rep in ppdb[word]:
		    if word in glove and rep in glove:
		        new_examples.append(make_new_ex(example, word, rep))
    return new_examples

def make_new_ex(example, original, replacement):
    premise = [replacement if word == original else word for word in example[0]]
    hypo = [replacement if word == original else word for word in example[1]]  
    return (premise, hypo, example[2])
		

def test_augmentation(glove, dev, ppdb_file):
    ppdb = paraphrase.load_parap(ppdb_file)
    aug = augmented_dataset(glove, dev, ppdb)
    return aug

def parapharse_models(glove, train, dev, ppdb_file):
    ppdb = paraphrase.load_parap(ppdb_file)
    aug = augmented_dataset(glove, train, ppdb)
    train_aug = train + aug
    
    models.train_model(train_aug, dev, glove, model_filename = 'models/train_aug')
    models.train_model(train, dev, glove, model_filename = 'models/train_noaug')    

def tune_model(observed_example, train_example, model, glove):
    class_arg = load_data.LABEL_LIST.index(observed_example[2])
    prem = " ".join(observed_example[0])
    hypo = " ".join(observed_example[1])
    print prem, hypo, observed_example[2], class_arg
    for i in range(30):
	probs = misc.predict_example(prem, hypo, model, glove)[0]
        print i, probs
        if probs.argmax() == class_arg:
            break
        models.update_model_once(model, glove, [train_example])
        

def generate_tautologies(dataset):
    unique = set()
    result = []
    for ex in dataset:
	premise = " ".join(ex[0])
        if  premise not in unique:
	    result.append((ex[0], ex[0], 'entailment'))
	    unique.add(premise)
    return result

def generate_contradictions(dataset):
    result = []
    for ex in dataset:
        if ex[2] == 'contradiction':
            result.append((ex[1],ex[0],ex[2]))
    return result

def generate_neutral(dataset):
    result = []
    for ex in dataset:
	if ex[2] == 'entailment':
            result.append((ex[1],ex[0],'neutral'))
    return result

def generate_all(dataset):
    return generate_tautologies(dataset) + generate_contradictions(dataset) + generate_neutral(dataset)     

def unknown_words_analysis(train, dev):
    train_words = set.union(*[set(ex[0]+ex[1]) for ex in train])
    indices = [[],[]]
    for i in range(len(dev)):
	diff = len(set(dev[i][0] + dev[i][1]) - train_words)
        if diff == 0:
	    indices[0].append(i)
	else:
	    indices[1].append(i)
    return indices

def color_analysis(dev):
    COLORS = set(['black', 'blue', 'orange', 'white', 'yellow', 'green', 'pink', 'purple', 'red', 'brown', 'gray', 'grey'])
    indices = [[],[]]
    for i in range(len(dev)):
        diff = len(set(dev[i][0] + dev[i][1]) & COLORS)
        if diff == 0:
            indices[0].append(i)
        else:
            indices[1].append(i)
    return indices

def mixture_experiments(train, dev, glove, splits = 5):
    for i in range(splits):
        model_name = 'mixture' + str(i)
        print 'Model', model_name
        model = models.init_model()
        div = len(train) / splits
        models.train_model(train[:i*div] + train[(i+1)*div:splits*div], dev, glove, model, 'models/' + model_name)

def extended_tautologies(train, dev, glove):
    augment_data = generate_all(train)
    from random import shuffle
    shuffle(augment_data)
    augment_weight = [0, 0.05, 0.15, 0.5]
    for w in augment_weight:
        new_train = train + augment_data[:int(len(train)*w)]
	str =  str(w).replace('.','')
        model = models.init_model()
	models.train_model(new_train, dev, glove, model = model, model_dir = 'models/aug' + w_str)
        
def test_tautologies(train, dev, glove, paths = ['aug0','aug005','aug015','aug05']):
    testsets = [dev, generate_tautologies(dev), generate_contradictions(dev), generate_neutral(dev)]
    names = ['dev' , 'ent', 'contr' ,'neu']
    for path in paths:
        print path
        model_path = misc.best_model_path('models/' + path)
        model = models.load_model(model_path)
        accs = [models.test_model(model, dataset, glove) for dataset in testsets]
	for name, dataset, acc in zip (names, testsets, accs):
	    print name, acc, len(dataset)
