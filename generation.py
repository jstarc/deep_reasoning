# -*- coding: utf-8 -*-

import sys
sys.path.append('../keras')
import os
import numpy as np


import load_data
import misc

from keras.utils.generic_utils import Progbar




    
def train_model_embed(train, dev, glove, model, model_dir = 'models/curr_model', nb_epochs = 20, batch_size = 64, hs=True, ci = True):
    X_dev_p, X_dev_h, y_dev = load_data.prepare_split_vec_dataset(dev, glove=glove)
    
    word_index = load_data.WordIndex(glove)
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], word_index.index)
            padded_p = load_data.pad_sequences(X_train_p, maxlen = PREM_LEN, dim = -1, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, maxlen = HYPO_LEN, dim = -1, padding = 'post')
            
            data = {'premise_input': padded_p, 'embed_input': np.expand_dims(np.array(train_index), axis=1), 'output' : padded_h}
            if ci:
                data['class_input'] = y_train
            if hs:
                data['train_input'] = padded_h
                data['output'] = np.ones((batch_size, HYPO_LEN, 1))
            
            #sw = (padded_h != 0).astype(float)
            #train_loss = float(model.train_on_batch(data, sample_weight={'output':sw})[0])
	    train_loss = float(model.train_on_batch(data)[0])
            p.add(len(train_index),[('train_loss', train_loss)])
        sys.stdout.write('\n')
        model.save_weights(model_dir + '/model~' + str(e))


def generation_predict_embed(model, word_index, batch, embed_indices, batch_size = 64, hs = True, class_indices = None):
    prem, hypo, y = load_data.prepare_split_vec_dataset(batch, word_index)
    X_p = load_data.pad_sequences(prem, maxlen=PREM_LEN, dim = -1)
    
    data = {'premise_input': X_p, 'embed_input': embed_indices[:,None]}
    
    if class_indices is not None:
      C = load_data.convert_to_one_hot(class_indices, 3)
      data['class_input'] = C
    
    if hs:
        data['train_input'] = np.zeros((batch_size, HYPO_LEN))
    
    model_pred = model.predict_on_batch(data)
    return model_pred['output']

def get_classes(preds):
   return np.argmax(preds, axis = 2)


def project(embed_sent, word_index):
    result = []
    ind = np.argmax(embed_sent, axis = 1)
    for i in ind:
        result.append(word_index.keys[i])
    return result

 
def get_word(array, glove, glove_mat):
    prod = np.dot(glove_mat, array)
    return glove.keys()[prod.argmax()]


def make_glove_mat(glove):
    glove_mat = np.array(glove.values())    
    glove_mat = glove_mat / np.linalg.norm(glove_mat, axis = 1)[:,None]
    return glove_mat




 
def test_genmodel(gen_model, train, dev, word_index, classify_model = None, glove_alter = None, batch_size = 64, ci = False):
   
    gens_count = 6
    dev_counts = 6
    gens = []
    for i in range(gens_count):
        creatives = np.random.random_integers(0, len(train), batch_size)
        preds = generation_predict_embed(gen_model, word_index.index, dev[:batch_size], creatives, class_indices = [i % 3] * batch_size)
        gens.append(get_classes(preds))

    for j in range(dev_counts):
        print " ".join(dev[j][0])
        print " ".join(dev[j][1])
        print "Generations: "
        for i in range(gens_count):
            gen_lex = gens[i][j]
            gen_str = word_index.print_seq(gen_lex)
            if ci:
               gen_str = load_data.LABEL_LIST[i%3][0] + ' ' + gen_str
            
            if classify_model != None and glove_alter != None: 
                probs = misc.predict_example(" ".join(dev[j][0]), gen_str, classify_model, glove_alter)
                print probs[0][0][i%3], gen_str
            else:
                print gen_str
        print
