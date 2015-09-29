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


def pad_sequences(sequences, maxlen=None, dim=1, dtype='float32',
    padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
    

train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
glove = load_data.import_glove('data/snli_vectors.txt')


model = Sequential()
model.add(LSTM(X_padded.shape[2], 128))
model.add(Dropout(0.2))
model.add(Dense(128, 3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print "Compiled"

X_padded = pad_sequences(X_dev, dim = 50)
model.fit(X_padded, y_dev, nb_epoch=20, validation_split=0.1, show_accuracy=True)
