# -*- coding: utf-8 -*-

import sys
sys.path.append('../seq2seq')
import os
import numpy as np
import csv
from keras.models import Sequential
from seq2seq.models import Seq2seq, SimpleSeq2seq, AttentionSeq2seq
import load_data
from keras.utils.generic_utils import Progbar
PREM_LEN = 50
HYPO_LEN = 30


def make_model(hidden_size = 10, embed_size = 50, batch_size = 64):
    
    model = AttentionSeq2seq(
        batch_input_shape = (batch_size, PREM_LEN, embed_size),
        input_dim = embed_size,
        hidden_dim=embed_size,
        output_dim=embed_size,
        output_length=HYPO_LEN,
        depth=1
    )

    #model.add(seq2seq, input_shape= (None, None, embed_size))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def generation_test(train, glove, model, batch_size = 64):
    mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
    p = Progbar(len(train))
    for i, train_index in mb:
        X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
        X_p = load_data.pad_sequences(X_prem, maxlen = PREM_LEN, dim = 50)
        X_h = load_data.pad_sequences(X_hypo, maxlen = HYPO_LEN, dim = 50)
        train_loss =  model.train_on_batch(X_p, X_h)[0]
        p.add(len(X_p),[('train_loss', train_loss)])
    
    
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
        #mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        mb = load_data.get_minibatches_idx_bucketing_both(train,([9,11,13,16,22],[6,7,8,10,13]), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
            padded_p = load_data.pad_sequences(X_train_p, maxlen = PREM_LEN, dim = embed_size)
            padded_h = load_data.pad_sequences(X_train_h, maxlen = HYPO_LEN, dim = embed_size)
            train_loss = float(model.train_on_batch(padded_p, padded_h)[0])
            p.add(len(padded_p),[('train_loss', train_loss)])
            iter = e * len(mb) + i + 1
            if iter % validation_freq == 0:
                print
                dev_loss = validate_model_generation(model, X_dev_p, X_dev_h, batch_size)
                print
                test_losses.append(dev_loss)
                stats.append([iter, train_loss, dev_loss])
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

def validate_model_generation(model, X_dev_p, X_dev_h, batch_size):
    dmb = load_data.get_minibatches_idx(len(X_dev_p), batch_size, shuffle=True)
    p = Progbar(len(X_dev_p))
    for i, dev_index in dmb:
        if len(dev_index) != batch_size:
            continue
        padded_p = load_data.pad_sequences(X_dev_p[dev_index], maxlen=PREM_LEN, dim = len(X_dev_p[0][0]))
        padded_h = load_data.pad_sequences(X_dev_h[dev_index], maxlen=HYPO_LEN, dim = len(X_dev_p[0][0]))
        loss, acc = model.test_on_batch(padded_p, padded_h, accuracy=True)
        p.add(len(padded_p),[('test_loss',loss)])
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
    
#glove_mat = glove_mat / np.linalg.norm(glove_mat, axis = 1)[:,None]

def transform_dataset(dataset):
    uniq = set()
    result = []
    for ex in dataset:
        if ex[2] == 'entailment' and :
            pass
