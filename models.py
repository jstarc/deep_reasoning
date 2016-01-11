# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:28:46 2016

@author: Janez
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json

import load_data
import numpy as np

def init_model(embed_size = 300, hidden_size = 100, lr = 0.001, dropout = 0.0, reg = 0.001):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape = (None, embed_size)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_size))
    model.add(Dense(3, W_regularizer=l2(reg)))
    model.add(Dropout(dropout))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr))

    return model
    
def train_model(train, dev, glove, model = init_model(), model_filename =  'models/curr_model', nb_epochs = 20, batch_size = 128):
    validation_freq = 1000
    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    test_losses = []
    worse_steps = 4  
    embed_size = X_dev[0].shape[1]
    for e in range(nb_epochs): 
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
	    X_train, y_train = load_data.prepare_vec_dataset([train[k] for k in train_index], glove)
	    X_padded = load_data.pad_sequences(X_train, dim = embed_size)
	    loss, acc = model.train_on_batch(X_padded, y_train, accuracy=True)
	    p.add(len(X_padded),[('train_loss',loss), ('train_acc', acc)])
	    iter = e * len(mb) + i + 1
            if iter % validation_freq == 0:
		print
		loss, acc = validate_model(model, X_dev, y_dev, batch_size, embed_size)
		print
		test_losses.append(loss)
		if (np.array(test_losses[-worse_steps:]) < min(test_losses)).all():
		    print test_loses
		    return
		else:
		    fn = model_filename + '~' + str(iter)
		    open(fn + '.json', 'w').write(model.to_json())
	            model.save_weights(fn + '.h5')
            
def validate_model(model, X_dev, y_dev, batch_size, embed_size):
    dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=True)
    p = Progbar(len(X_dev))
    for i, dev_index in dmb:
        X_padded = load_data.pad_sequences(X_dev[dev_index], dim = embed_size)
        loss, acc = model.test_on_batch(X_padded, y_dev[dev_index], accuracy=True)
        p.add(len(X_padded),[('test_loss',loss), ('test_acc', acc)])
    loss = p.sum_values['test_loss'][0] / p.sum_values['test_loss'][1]
    acc = p.sum_values['test_acc'][0] / p.sum_values['test_acc'][1]
    return loss, acc

def update_model_once(model, glove, train_data, embed_size):
    X_train, y_train = load_data.prepare_vec_dataset(train_data, glove)
    X_padded = load_data.pad_sequences(X_train, dim = embed_size)
    model.train_on_batch(X_padded, y_train, accuracy=True)    
        
def load_model(model_filename):
    model = model_from_json(open(model_filename + '.json').read())
    model.load_weights(model_filename + '.h5')
    return model

     
def test_model(model, dev, glove, batch_size = 100, return_probs = False):
    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=False)
    y_pred = np.zeros((len(y_dev), 3))
    for i, dev_index in dmb:
        X_padded = load_data.pad_sequences(X_dev[dev_index], dim = len(X_dev[0][0]))
        y_pred[dev_index] = model.predict_on_batch(X_padded)

    acc =  np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_dev, axis=1)) / float(len(y_pred))
    if return_probs:
	return acc, y_pred
    else:
	return acc
