# -*- coding: utf-8 -*-

import sys
sys.path.append('../seq2seq')
sys.path.append('../keras')
import os
import numpy as np
import csv
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten
from keras.layers.core import Dense, Merge, RepeatVector, TimeDistributedDense

 
from seq2seq.models import Seq2seq, SimpleSeq2seq, AttentionSeq2seq
import load_data
from keras.utils.generic_utils import Progbar
PREM_LEN = 22
HYPO_LEN = 12


def make_model(hidden_size = 10, embed_size = 50, batch_size = 64):
    
    batch_input_shape = (batch_size, PREM_LEN, embed_size)
    
    model = Sequential()
    seq2seq = AttentionSeq2seq(
        batch_input_shape = batch_input_shape,
        input_dim = embed_size,
        hidden_dim=embed_size,
        output_dim=embed_size,
        output_length=HYPO_LEN,
        depth=1,
        bidirectional=False,
    )

    model.add(seq2seq)
    model.compile(loss='mse', optimizer='rmsprop', sample_weight_mode="temporal")
    return model
    
def make_embed_model(vocab_size, hidden_size = 10, embed_size = 50, batch_size = 64):
    
    batch_input_shape = (batch_size, PREM_LEN, embed_size)
    
    em_model = Sequential()    
    em_model.add(Embedding(vocab_size, embed_size, input_length = 1))
    em_model.add(Flatten())
    #em_model.add(Dense(embed_size))
    em_model.add(RepeatVector(PREM_LEN))
    
    seq2seq = AttentionSeq2seq(
        batch_input_shape = batch_input_shape,
        input_dim = embed_size * 2,
        hidden_dim=embed_size,
        output_dim=embed_size,
        output_length=HYPO_LEN,
        depth=1,
        bidirectional=False,
    )
    
    graph = Graph()
    graph.add_input(name='premise_input', batch_input_shape=batch_input_shape)
    graph.add_input(name='embed_input', batch_input_shape=(batch_size,1), dtype='int')
    
    
    graph.add_node(em_model, name='em_model', input='embed_input')
    
    graph.add_node(seq2seq, name='seq2seq', inputs=['premise_input', 'em_model'], merge_mode='concat')
    graph.add_output(name='output', input='seq2seq')
    
    graph.compile(loss={'output':'mse'}, optimizer='rmsprop')
    return graph

def generation_test(train, glove, model, batch_size = 64):
    mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
    p = Progbar(len(train))
    for i, train_index in mb:
        X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
        X_p = load_data.pad_sequences(X_prem, maxlen = PREM_LEN, dim = 50)
        X_h = load_data.pad_sequences(X_hypo, maxlen = HYPO_LEN, dim = 50)
        train_loss =  model.train_on_batch(X_p, X_h)[0]
        p.add(len(X_p),[('train_loss', train_loss)])
        
def generation_embded_test(train, glove, model, batch_size = 64):
    batch = np.arange(batch_size)
    X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in batch], glove)
    X_p = load_data.pad_sequences(X_prem, maxlen = PREM_LEN, dim = 50)
    X_h = load_data.pad_sequences(X_hypo, maxlen = HYPO_LEN, dim = 50)
    data = {'premise_input': X_p, 'embed_input': np.expand_dims(np.array(batch), axis=1), 'output' : X_h}        
    return model.train_on_batch(data)
        #p.add(len(X_p),[('train_loss', train_loss)])
    
    
def train_model_generation(train, dev, glove, model, model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 64, worse_steps = 5):
    validation_freq = 1000
    X_dev_p, X_dev_h, y_dev = load_data.prepare_split_vec_dataset(dev, glove)
    test_losses = []
    stats = [['iter', 'train_loss', 'dev_loss']]
    exit_loop = False
    embed_size = X_dev_p[0].shape[1]

    if not os.path.exists(model_dir):
         os.makedirs(model_dir)

    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        #mb = load_data.get_minibatches_idx_bucketing_both(train,([9,11,13,16,22],[6,7,8,10,13]), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
            padded_p = load_data.pad_sequences(X_train_p, maxlen = PREM_LEN, dim = embed_size, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, maxlen = HYPO_LEN, dim = embed_size, padding = 'post')
            sw = (np.sum(padded_h, axis = 2) != 0).astype(float)
            train_loss = float(model.train_on_batch(padded_p, padded_h, sample_weight = sw)[0])
            p.add(len(train_index),[('train_loss', train_loss)])
            iter = e * len(mb) + i + 1
            #if iter % validation_freq == 0:
        sys.stdout.write('\n')
        dev_loss = validate_model_generation(model, X_dev_p, X_dev_h, glove, batch_size)
        sys.stdout.write('\n\n')
        test_losses.append(dev_loss)
        stats.append([iter,  p.sum_values['train_loss'][0] / p.sum_values['train_loss'][1], dev_loss])
        if (np.array(test_losses[-worse_steps:]) > min(test_losses)).all():
            exit_loop = True
            break
        else:
            fn = model_dir + '/model' '~' + str(iter)
            open(fn + '.json', 'w').write(model.to_json())
            model.save_weights(fn + '.h5')
        if exit_loop:
            break
    with open(model_dir + '/stats.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(stats)
        
def train_model_embed(train, dev, glove, model, model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 64, worse_steps = 5):
    #validation_freq = 1000
    X_dev_p, X_dev_h, y_dev = load_data.prepare_split_vec_dataset(dev, glove)
    #test_losses = []
    #stats = [['iter', 'train_loss', 'dev_loss']]
    #exit_loop = False
    embed_size = X_dev_p[0].shape[1]

    if not os.path.exists(model_dir):
         os.makedirs(model_dir)

    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        #mb = load_data.get_minibatches_idx_bucketing_both(train,([9,11,13,16,22],[6,7,8,10,13]), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
            padded_p = load_data.pad_sequences(X_train_p, maxlen = PREM_LEN, dim = embed_size, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, maxlen = HYPO_LEN, dim = embed_size, padding = 'post')
            data = {'premise_input': padded_p, 'embed_input': np.expand_dims(np.array(train_index), axis=1), 'output' : padded_h}
            train_loss = float(model.train_on_batch(data)[0])
            #sw = (np.sum(padded_h, axis = 2) != 0).astype(float)
            #train_loss = float(model.train_on_batch(padded_p, padded_h, sample_weight = sw)[0])
            p.add(len(train_index),[('train_loss', train_loss)])
            #iter = e * len(mb) + i + 1
            #if iter % validation_freq == 0:
        #sys.stdout.write('\n')
        #dev_loss = validate_model_generation(model, X_dev_p, X_dev_h, glove, batch_size)
        #sys.stdout.write('\n\n')
        #test_losses.append(dev_loss)
        #stats.append([iter,  p.sum_values['train_loss'][0] / p.sum_values['train_loss'][1], dev_loss])
        #if (np.array(test_losses[-worse_steps:]) > min(test_losses)).all():
        #    exit_loop = True
        #    break
        #else:
        #    fn = model_dir + '/model' '~' + str(iter)
        #    open(fn + '.json', 'w').write(model.to_json())
        #    model.save_weights(fn + '.h5')
        #if exit_loop:
        #    break
    #with open(model_dir + '/stats.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(stats)

def validate_model_generation(model, X_dev_p, X_dev_h, glove, batch_size):
    glove_mat = np.array(glove.values())
    glove_mat = glove_mat / np.linalg.norm(glove_mat, axis = 1)[:,None]    
    dmb = load_data.get_minibatches_idx(len(X_dev_p), batch_size, shuffle=True)
    p = Progbar(len(X_dev_p))
    for i, dev_index in dmb:
        if len(dev_index) != batch_size:
            continue
        padded_p = load_data.pad_sequences(X_dev_p[dev_index], maxlen=PREM_LEN, dim = len(X_dev_p[0][0]), padding = 'pre')
        padded_h = load_data.pad_sequences(X_dev_h[dev_index], maxlen=HYPO_LEN, dim = len(X_dev_p[0][0]), padding = 'post')
        sw = (np.sum(padded_h, axis = 2) != 0).astype(float) 
        loss, acc = model.test_on_batch(padded_p, padded_h, accuracy=True,sample_weight = sw)
        p.add(len(dev_index),[('test_loss',loss)])
    loss = p.sum_values['test_loss'][0] / p.sum_values['test_loss'][1]
    return loss


def generation_predict(model, glove, premise, batch_size = 64):
    X_p = load_data.load_word_vecs(premise, glove)
    X_p = load_data.pad_sequences([X_p], maxlen=PREM_LEN, dim = len(X_p[0]))
    X = np.zeros((batch_size, X_p.shape[1], X_p.shape[2]))
    X[0] = X_p[0]
    model_pred = model.predict_on_batch(X)
    return model_pred[0][0]

def project(embed_sent, glove, glove_mat):
    result = []
    for e in embed_sent:
        result.append(get_word(e, glove, glove_mat))
    return result

def rank_sent(premise, hypothesis, model, glove, glove_mat):
    result = []
    y_pred = generation_predict(model, glove, premise)
    for i in range(len(hypothesis)):
        result.append(get_rank(y_pred[i], hypothesis[i], glove, glove_mat))
    return result    

def get_word(array, glove, glove_mat):
    prod = np.dot(glove_mat, array)
    return glove.keys()[prod.argmax()]

def get_rank(array, word, glove, glove_mat):
    prod = np.dot(glove_mat, array)
    ind = glove.keys().index(word)
    return len(glove) - prod.argsort().argsort()[ind]


def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

#glove_mat = np.array(glove.values())    
#glove_mat = glove_mat / np.linalg.norm(glove_mat, axis = 1)[:,None]

def transform_dataset(dataset, class_str = 'entailment', max_prem_len = sys.maxint, max_hypo_len = sys.maxint):
    uniq = set()
    result = []
    for ex in dataset:
        prem_str = " ".join(ex[0])
        if ex[2] == class_str and prem_str not in uniq and len(ex[0]) <= max_prem_len and len(ex[1]) <= max_hypo_len:
            result.append(ex)
            uniq.add(prem_str)
    return result

def add_eol_to_hypo(dataset):
    for ex in dataset:
        if ex[1][-1] != '.':
            ex[1].append('.')
        
    

def word_overlap(premise, hypo):
    return len([h for h in hypo if h in premise]) / float(len(hypo))    

def predict_lexicalize(example, model, glove, glove_mat):
    pred = generation_predict(model, glove, example[0])
    print " ".join(example[0])
    print " ".join(example[1])
    print " ".join(project(pred, glove, glove_mat))           
