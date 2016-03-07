# -*- coding: utf-8 -*-

import sys
sys.path.append('../keras')

from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.models import Sequential, Graph
from keras import backend as K
from keras.layers.core import Lambda
from keras.utils.generic_utils import Progbar

import numpy as np
import generation
import load_data
from keras.objectives import binary_crossentropy



def make_adverse_model(generative_model, glove, embed_size = 50, batch_size = 64,
                       hypo_len = 12):
    discriminator = Sequential()
    discriminator.add(generation.make_fixed_embeddings(glove, hypo_len))
    discriminator.add(LSTM(embed_size))
    discriminator.add(Dense(1, activation='sigmoid'))    
    
    
    graph = Graph()
    #graph.add_input(name='premise_input', batch_input_shape=batch_input_shape)
    #graph.add_input(name='embed_input', batch_input_shape=(batch_size,1), dtype='int')
    graph.add_node(generative_model, name = 'generative')#, inputs=['premise_input', 'embed_input'])
    graph.add_input(name='training_hypo', batch_input_shape=(batch_size, hypo_len), dtype = 'int')
    graph.add_shared_node(discriminator, name='shared', inputs=['training_hypo', 'generative'], merge_mode='join')
    
    def margin_opt(inputs):
        print(inputs)
        assert len(inputs) == 2, ('Margin Output needs '
                              '2 inputs, %d given' % len(inputs))
        u, v = inputs.values()
        return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))
    graph.add_node(Lambda(margin_opt), name = 'output2', input='shared', create_output = True)
    graph.cache_enabled = False
    graph.compile(loss={'output2':'mse'}, optimizer='adam')
    return graph

def make_basic_adverse(glove, embed_size = 50, compile=False, hypo_len = 12):
    discriminator = Sequential()
    discriminator.add(generation.make_fixed_embeddings(glove, hypo_len))
    discriminator.add(LSTM(embed_size))
    discriminator.add(Dense(1, activation='sigmoid'))
    if compile:
        discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

def make_adverse_model2(glove, embed_size = 50, batch_size = 64, hypo_len = 12):
    discriminator = make_basic_adverse(glove, embed_size)
    
    graph = Graph()
    graph.add_input(name='train_hypo', batch_input_shape=(batch_size, hypo_len), dtype ='int')
    graph.add_input(name='gen_hypo', batch_input_shape=(batch_size, hypo_len), dtype ='int')
    graph.add_shared_node(discriminator, name='shared', inputs=['train_hypo', 'gen_hypo'], merge_mode='join')
    
    def margin_opt(inputs):
        print(inputs)
        assert len(inputs) == 2, ('Margin Output needs '
                              '2 inputs, %d given' % len(inputs))
        u, v = inputs.values()
        return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))
    graph.add_node(Lambda(margin_opt), name = 'output2', input='shared', create_output = True)
    graph.compile(loss={'output2':'mse'}, optimizer='adam')
    return graph

    

def adverse_model_train(train, ad_model, gen_model, word_index, glove, nb_epochs = 20, batch_size=64, ci=False):
    
    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(2 * len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            class_indices = [i % 3] * batch_size if ci else None
            X, y = adverse_batch([train[k] for k in train_index], word_index, gen_model, len(train), class_indices = class_indices)              
            loss = ad_model.train_on_batch(X, y)[0]
            p.add(len(X),[('train_loss', loss)])

def test_adverse(dev, ad_model, gen_model, word_index, glove, train_len, batch_size=64):
    mb = load_data.get_minibatches_idx(len(dev), batch_size, shuffle=False)
    p = Progbar(len(dev) * 2)
    for i, train_index in mb:
        if len(train_index) != batch_size:
            continue
        X, y = adverse_batch([dev[k] for k in train_index], word_index, gen_model, train_len)
        pred = ad_model.predict_on_batch(X)[0].flatten()
        loss = binary_crossentropy(y, pred).eval()[0]
        acc = sum(np.abs(y - pred) < 0.5) / float(len(y))
        p.add(len(X),[('test_loss', loss), ('test_acc', acc)])

def adverse_generate(gen_model, ad_model, train, word_index, threshold = 0.95, batch_size = 64):
    mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
    results = []
    for i, train_index in mb:
        if len(train_index) != batch_size:
            continue    
        orig_batch = [train[k] for k in train_index]
        probs = generation.generation_predict_embed(gen_model, word_index.index, orig_batch,
                     np.random.random_integers(0, len(train), len(orig_batch)))
        gen_batch = generation.get_classes(probs)
        preds = ad_model.predict_on_batch(gen_batch)[0].flatten()
        zipped = zip(preds, [word_index.print_seq(gen) for gen in gen_batch])
        results += [el for el in zipped if el[0] > threshold]
        if len(results) > 64:
            print (i + 1) * batch_size 
            return results
            
def adverse_batch(orig_batch, word_index, gen_model, train_len, class_indices = None, hypo_len = 12):
    probs = generation.generation_predict_embed(gen_model, word_index.index, orig_batch,
                     np.random.random_integers(0, train_len, len(orig_batch)), class_indices=class_indices)
    gen_batch = generation.get_classes(probs)

    _, X_hypo, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
    train_batch = load_data.pad_sequences(X_hypo, maxlen = hypo_len, dim = -1, padding = 'post')

    X = np.concatenate([gen_batch, train_batch])
    y = np.concatenate([np.zeros(len(gen_batch)), np.ones(len(train_batch))])
    return X,y
    

