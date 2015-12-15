# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:43:55 2015

@author: Janez
"""

import sys
sys.path.append('../keras')

import load_data
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json



if __name__ == "__main__":
    train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
    glove = load_data.import_glove('data/snli_vectors_300.txt')


def train_model(train, dev, glove, model_filename =  'models/curr_model'):
    EMBED_SIZE = 300
    HIDDEN_SIZE = 100

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape = (None, EMBED_SIZE)))
    model.add(LSTM(HIDDEN_SIZE))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print "Compiled"

    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)

    nb_epochs = 20
    batch_size = 128

    best_acc = 0.0

    for e in range(nb_epochs): 
        print "Epoch ", e,
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
	    X_train, y_train = load_data.prepare_vec_dataset([train[k] for k in train_index], glove)
	    X_padded = load_data.pad_sequences(X_train, dim = EMBED_SIZE)
	    loss, acc = model.train_on_batch(X_padded, y_train, accuracy=True)
	    p.add(len(X_padded),[('train_loss',loss), ('train_acc', acc)])
	dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=True)
	p = Progbar(len(X_dev))
	for i, dev_index in dmb:
	    X_padded = load_data.pad_sequences(X_dev[dev_index], dim = EMBED_SIZE)
	    loss, acc = model.test_on_batch(X_padded, y_dev[dev_index], accuracy=True)
	    p.add(len(X_padded),[('test_loss',loss), ('test_acc', acc)])
	acc = p.sum_values['test_acc'][0] / p.sum_values['test_acc'][1]
	if acc > best_acc:
	    best_acc = acc
	else:
	    break
	open(model_filename + str(e) + '.json', 'w').write(model.to_json())
	model.save_weights(model_filename + str(e) + '.h5')




def load_model(model_filename):
    model = model_from_json(open(model_filename + '.json').read())
    model.load_weights(model_filename + '.h5')
    return model

     

def test_model(model, dev, glove, batch_size = 100):
    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=False)
    y_pred = np.zeros((len(y_dev), 3))
    for i, dev_index in dmb:
        X_padded = load_data.pad_sequences(X_dev[dev_index], dim = len(X_dev[0][0]))
        y_pred[dev_index] = model.predict_on_batch(X_padded)


    y_diff = y_dev - y_pred
    class_max = np.max(y_diff, axis=1)

    display = 100
    most_wrong = class_max.argsort()[-display:]
    print np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_dev, axis=1)) / float(len(y_pred))
    return y_pred


def test_model2(model, dev, glove):
    from misc import predict_example
    tp = 0
    for ex in dev:
	probs = predict_example(" ".join(ex[0]), " ".join(ex[1]), model, glove)
	label = load_data.LABEL_LIST[np.argmax(probs)]
	if label == ex[2]:
	   tp +=1
    print tp / float(len(dev))
   

