# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:28:46 2016

@author: Janez
"""
import sys
sys.path.append('../keras')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json
from keras.objectives import categorical_crossentropy as cc

import load_data
import numpy as np
import os
import csv


#import sys
#sys.path.append('../seq2seq')
#from seq2seq.layers.state_transfer_lstm import StateTransferLSTM
#from keras.models import Graph
#
#def alter_model(embed_size = 300, hidden_size = 100, batch_size = 128):
#    premise_layer = StateTransferLSTM(output_dim=hidden_size, state_input=False, input_shape=(batch_size, embed_size))
#    hypo_layer = StateTransferLSTM(output_dim=hidden_size, state_input=True, input_shape=(batch_size, embed_size))
#    premise_layer.broadcast_state(hypo_layer)
#       
#    graph = Graph()
#    graph.add_input(name='premise_input', input_shape= (batch_size, embed_size))
#    graph.add_input(name='hypo_input', input_shape= (batch_size, embed_size))
#    graph.add_node(premise_layer, name='premise', input='premise_input')
#    graph.add_node(hypo_layer, name='hypo', input='hypo_input')
#    graph.add_node(Dense(3), name='dense', input='hypo')
#    graph.add_node(Activation('softmax'), name='softmax', input='dense')
#    graph.add_output(name='output', input='softmax')
#    graph.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam())
#    return graph

def init_model(embed_size = 300, hidden_size = 100, lr = 0.001, dropout = 0.0, reg = 0.001):
    model = Sequential()
    #model.add(Masking(mask_value=0, input_shape = (None, embed_size)))
    model.add(Dropout(dropout, input_shape = (None, embed_size)))
    model.add(LSTM(hidden_size))
    model.add(Dense(3, W_regularizer=l2(reg)))
    model.add(Dropout(dropout))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr))

    return model
    
def train_model(train, dev, glove, model = init_model(), model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 128, worse_steps = 4):
    validation_freq = 1000
    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    test_losses = []
    stats = [['iter', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']]
    exit_loop = False
    embed_size = X_dev[0].shape[1]
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    for e in range(nb_epochs): 
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        #mb = load_data.get_minibatches_idx_bucketing([len(ex[0]) + len(ex[1]) for ex in train], batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            X_train, y_train = load_data.prepare_vec_dataset([train[k] for k in train_index], glove)
            X_padded = load_data.pad_sequences(X_train, dim = embed_size)
            train_loss, train_acc = model.train_on_batch(X_padded, y_train, accuracy=True)
            p.add(len(X_padded),[('train_loss', train_loss), ('train_acc', train_acc)])
            iter = e * len(mb) + i + 1
            if iter % validation_freq == 0:
                print
                dev_loss, dev_acc = validate_model(model, X_dev, y_dev, batch_size)
                print
                test_losses.append(dev_loss)
                stats.append([iter, train_loss, train_acc, dev_loss, dev_acc])
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

def train_model_graph(train, dev, glove, model = init_model(), model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 128, worse_steps = 4):
    validation_freq = 1000
    #X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    X_dev_p, X_dev_h, y_dev = load_data.prepare_split_vec_dataset(dev, glove)
    test_losses = []
    stats = [['iter', 'train_loss', 'dev_loss', 'dev_acc']]
    exit_loop = False
    embed_size = X_dev_p[0].shape[1]
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    
    for e in range(nb_epochs): 
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        #mb = load_data.get_minibatches_idx_bucketing([len(ex[0]) + len(ex[1]) for ex in train], batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            X_train_p, X_train_h, y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
            padded_p = load_data.pad_sequences(X_train_p, dim = embed_size)
            padded_h = load_data.pad_sequences(X_train_h, dim = embed_size)
            data = {'premise_input': padded_p, 'hypo_input': padded_h, 'output' : y_train}
            train_loss = float(model.train_on_batch(data)[0])
            p.add(len(padded_p),[('train_loss', train_loss)])
            iter = e * len(mb) + i + 1
            if iter % validation_freq == 0:
                print
                dev_loss, dev_acc = validate_model_graph(model, X_dev_p, X_dev_h, y_dev, batch_size)
                print
                test_losses.append(dev_loss)
                stats.append([iter, train_loss, dev_loss, dev_acc])
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
 
 
 
def validate_model_graph(model, X_dev_p, X_dev_h, y_dev, batch_size):
    dmb = load_data.get_minibatches_idx(len(X_dev_p), batch_size, shuffle=True)
    p = Progbar(len(X_dev_p))
    for i, dev_index in dmb:
        padded_p = load_data.pad_sequences(X_dev_p[dev_index], dim = len(X_dev_p[0][0]))
        padded_h = load_data.pad_sequences(X_dev_h[dev_index], dim = len(X_dev_p[0][0]))
        data = {'premise_input': padded_p, 'hypo_input': padded_h}
        y_pred = model.predict(data)
        loss = -np.sum(y_dev[dev_index] * np.log(y_pred)) / float(len(y_pred))
        acc =  np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_dev[dev_index], axis=1)) / float(len(y_pred))
        p.add(len(padded_p),[('test_loss', loss), ('test_acc', acc)])
    loss = p.sum_values['test_loss'][0] / p.sum_values['test_loss'][1]
    acc = p.sum_values['test_acc'][0] / p.sum_values['test_acc'][1]
    return loss, acc
    
def validate_model(model, X_dev, y_dev, batch_size):
    dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=True)
    p = Progbar(len(X_dev))
    for i, dev_index in dmb:
        X_padded = load_data.pad_sequences(X_dev[dev_index], dim = len(X_dev[0][0]))
        loss, acc = model.test_on_batch(X_padded, y_dev[dev_index], accuracy=True)
        p.add(len(X_padded),[('test_loss',loss), ('test_acc', acc)])
    loss = p.sum_values['test_loss'][0] / p.sum_values['test_loss'][1]
    acc = p.sum_values['test_acc'][0] / p.sum_values['test_acc'][1]
    return loss, acc

def update_model_once(model, glove, train_data):
    X_train, y_train = load_data.prepare_vec_dataset(train_data, glove)
    X_padded = load_data.pad_sequences(X_train, dim = len(X_train[0][0]))
    model.train_on_batch(X_padded, y_train, accuracy=True)    
        
def load_model(model_filename):
    model = model_from_json(open(model_filename + '.json').read())
    model.load_weights(model_filename + '.h5')
    return model

     
def test_model(model, dev, glove, batch_size = 100, return_probs = False):
    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=False)
    #dmb = load_data.get_minibatches_idx_bucketing([len(ex[0]) + len(ex[1]) for ex in dev], batch_size, shuffle=True)
    y_pred = np.zeros((len(y_dev), 3))
    for i, dev_index in dmb:
        X_padded = load_data.pad_sequences(X_dev[dev_index], dim = len(X_dev[0][0]))
        y_pred[dev_index] = model.predict_on_batch(X_padded)

    acc =  np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_dev, axis=1)) / float(len(y_pred))
    if return_probs:
        return acc, y_pred
    else:
        return acc

