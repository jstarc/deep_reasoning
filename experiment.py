# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:43:55 2015

@author: Janez
"""
import load_data
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils.generic_utils import Progbar




train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
glove = load_data.import_glove('data/snli_vectors.txt')

X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)


X_train, y_train = load_data.prepare_vec_dataset(train, glove)

#X_train = X_train[:200000]
#y_train = y_train[:200000]


model = Sequential()
model.add(Masking(mask_value=0.))
model.add(LSTM(50, 128))
model.add(Dropout(0.2))
model.add(Dense(128, 3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print "Compiled"



nb_epochs = 3
batch_size = 128


for e in range(nb_epochs): 
    print "Epoch ", e,
    mb = load_data.get_minibatches_idx(len(X_train), batch_size, shuffle=True)
    p = Progbar(len(X_train))
    for i, train_index in mb:
	X_padded = load_data.pad_sequences(X_train[train_index], dim = 50)
	loss, acc = model.train_on_batch(X_padded, y_train[train_index], accuracy=True)
	p.add(len(X_padded),[('train_loss',loss), ('train_acc', acc)])
    dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=True)
    p = Progbar(len(X_dev))
    for i, dev_index in dmb:
	X_padded = load_data.pad_sequences(X_dev[dev_index], dim = 50)
	loss, acc = model.test_on_batch(X_padded, y_dev[dev_index], accuracy=True)
	p.add(len(X_padded),[('test_loss',loss), ('test_acc', acc)])

     

#X_padded = load_data.pad_sequences(X_train, dim = 50)
#model.fit(X_padded, y_train, nb_epoch=nb_epochs, batch_size=batch_size, validation_split=0.1, show_accuracy=True)


#dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=True)
#y_pred = np.zeros(len(y_dev, 3)
#for i, dev_index in dmb:
#    X_padded = load_data.pad_sequences(X_dev[dev_index], dim = 50)
#     y_pred[]model.predict_on_batch(X_dev[dev_index])
