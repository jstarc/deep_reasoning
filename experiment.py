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
from keras.regularizers import l2
from keras.optimizers import Adam

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
	model = init_model(embed_size, hidden_size, params[0], params[1], params[2])
	train_model(train, dev, glove, model, 'models/' + filename)    
	

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

    X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
    best_acc = 0.0
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
	dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=True)
	p = Progbar(len(X_dev))
	for i, dev_index in dmb:
	    X_padded = load_data.pad_sequences(X_dev[dev_index], dim = embed_size)
	    loss, acc = model.test_on_batch(X_padded, y_dev[dev_index], accuracy=True)
	    p.add(len(X_padded),[('test_loss',loss), ('test_acc', acc)])
	acc = p.sum_values['test_acc'][0] / p.sum_values['test_acc'][1]
	if acc > best_acc:
	    best_acc = acc
	else:
	    break
	open(model_filename + '~' + str(e) + '.json', 'w').write(model.to_json())
	model.save_weights(model_filename + '~' +str(e) + '.h5')




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

    #return y_pred


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
    extless = set([file.split('.')[0] for file in files]) - set([''])
    epoch_less = set([file.split('~')[0] for file in extless])
    for model_short in epoch_less:
	epoch = 0	
	if model_short in extless:
	    modelname = model_short
	else: 
	    while True:
		modelname_temp = model_short + '~' + str(epoch)
		if modelname_temp not in extless:
		    break
		modelname = modelname_temp
		epoch += 1
	
	print modelname
	model = load_model(folder + modelname)
	dev_acc = test_model(model, dev, glove)
        test_acc = test_model(model, test, glove)
	print "Dev:", '{0:.2f}'.format(dev_acc * 100), "Test_acc:", '{0:.2f}'.format(test_acc * 100)
	print 
 
    
    
	
	
