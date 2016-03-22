# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:28:46 2016

@author: Janez
"""
import sys
sys.path.append('../keras')


from keras.utils.generic_utils import Progbar
from keras.models import model_from_json


import load_data
import numpy as np


    



def update_model_once(model, glove, train_data):
    X_train, y_train = load_data.prepare_vec_dataset(train_data, glove=glove)
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

