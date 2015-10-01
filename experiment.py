# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:43:55 2015

@author: Janez
"""
import load_data
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM




#train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
#glove = load_data.import_glove('data/snli_vectors.txt')

#X_dev, y_dev = load_data.prepare_vec_dataset(dev, glove)
#X_padded = load_data.pad_sequences(X_dev, dim = 50)

#X_train, y_train = load_data.prepare_vec_dataset(train, glove)
#X_padded = load_data.pad_sequences(X_train, dim = 50)

model = Sequential()
model.add(LSTM(50, 128))
model.add(Dropout(0.2))
model.add(Dense(128, 3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print "Compiled"

for i in np.arange(len(X_train)):
    model.train_on_batch(np.expand_dims(X_train[i], axis=0), np.expand_dims(y_train[i], axis=0))
    if i % 1000 == 0:
	print i 

#model.fit(X_padded, y_train, nb_epoch=20, validation_split=0.1, show_accuracy=True)
