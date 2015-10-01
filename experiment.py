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


dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=False)
y_pred = np.zeros((len(y_dev), 3))
for i, dev_index in dmb:
    X_padded = load_data.pad_sequences(X_dev[dev_index], dim = 50)
    y_pred[dev_index] = model.predict_on_batch(X_padded)


y_diff = y_dev - y_pred
class_max = np.max(y_diff, axis=1)

display = 10
most_wrong = class_max.argsort()[-display:]

############################END
    
 print "Loading dev data"
    dev = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_dev.jsonl'))
    print "Loading test data"
    test = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_test.jsonl'))
    print "Data loaded"
    return train, dev, test

#repackage_glove('E:\\Janez\\Data\\vectors.6B.50d.txt', 'E:\\Janez\\Data\\snli_vectors.txt', 'E:\\Janez\\Data\\snli_1.0\\')


def prepare_vec_dataset(dataset, glove):
    X = []   
    y = []
    for example in dataset:
        concat = example[0] + ["--"] + example[1]
        X.append(load_word_vecs(concat, glove, 50))
        y.append(LABEL_LIST.index(example[2]))
    one_hot_y = np.zeros((len(y), len(LABEL_LIST)))
    one_hot_y[np.arange(len(y)), y] = 1
    return np.array(X), one_hot_y
    
def load_word_vec(token, glove, dim = 50):
    if token not in glove:
	glove[token] = np.random.uniform(-0.05, 0.05, dim)    
    return glove[token]
    
def load_word_vecs(token_list, glove, dim):
    return np.array([load_word_vec(x, glove, dim) for x in token_list])        
        
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

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_s# -*- coding: utf-8 -*-
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


dmb = load_data.get_minibatches_idx(len(X_dev), batch_size, shuffle=False)
y_pred = np.zeros((len(y_dev), 3))
for i, dev_index in dmb:
    X_padded = load_data.pad_sequences(X_dev[dev_index], dim = 50)
    y_pred[dev_index] = model.predict_on_batch(X_padded)


y_diff = y_dev - y_pred
class_max = np.max(y_diff, axis=1)

display = 10
most_wrong = class_max.argsort()[-display:]






        


    



        
        
        



