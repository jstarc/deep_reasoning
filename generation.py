# -*- coding: utf-8 -*-

import sys
sys.path.append('../seq2seq')
sys.path.append('../keras')
import os
import numpy as np
import csv
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten, Lambda
from keras.layers.core import Dense, RepeatVector, TimeDistributedDense, Activation

from keras import backend as K

 
from seq2seq.models import AttentionSeq2seq
import load_data
import misc
from hierarchical_softmax import HierarchicalSoftmax
from hierarchical_softmax import hs_categorical_crossentropy
from keras.utils.generic_utils import Progbar
PREM_LEN = 22
HYPO_LEN = 12

def make_fixed_embeddings(glove, seq_len):
    glove_mat = np.array(glove.values())
    return Embedding(input_dim = glove_mat.shape[0], output_dim = glove_mat.shape[1], 
                       weights = [glove_mat], trainable = False, input_length  = seq_len)
            
def make_embed_model(examples, glove, hidden_size = 10, embed_size = 50, batch_size = 64, hs = True, ci = True):
    
    batch_input_shape = (batch_size, PREM_LEN, embed_size)
    
    em_model = Sequential()    
    em_model.add(Embedding(examples, embed_size, input_length = 1, batch_input_shape=(batch_size,1)))
    em_model.add(Flatten())
    em_model.add(Dense(embed_size))
    em_model.add(RepeatVector(PREM_LEN))
    
    input_dim = embed_size * 2
    if ci:
        input_dim += 3
    seq2seq = AttentionSeq2seq(
        batch_input_shape = batch_input_shape,
        input_dim = input_dim,
        hidden_dim=embed_size,
        output_dim=embed_size,
        output_length=HYPO_LEN,
        depth=1,
        bidirectional=False,
    )

    class_model = Sequential()
    class_model.add(RepeatVector(PREM_LEN))
    
    graph = Graph()
    graph.add_input(name='premise_input', batch_input_shape=(batch_size, PREM_LEN), dtype = 'int')
    graph.add_node(make_fixed_embeddings(glove, PREM_LEN), name = 'word_vec', input='premise_input')
    
    graph.add_input(name='embed_input', batch_input_shape=(batch_size,1), dtype='int')
    graph.add_node(em_model, name='em_model', input='embed_input')
    
    seq_inputs = ['word_vec', 'em_model']
    
    if ci:
        graph.add_input(name='class_input', batch_input_shape=(batch_size,3))
        graph.add_node(class_model, name='class_model', input='class_input')
        seq_inputs += ['class_model']
   
    graph.add_node(seq2seq, name='seq2seq', inputs=seq_inputs, merge_mode='concat')
    
    if hs: 
        graph.add_input(name='train_input', batch_input_shape=(batch_size, HYPO_LEN), dtype='int32')
        graph.add_node(HierarchicalSoftmax(len(glove), input_dim = embed_size, input_length = HYPO_LEN), 
                   name = 'softmax', inputs=['seq2seq','train_input'], 
                   merge_mode = 'join')
    else:
        graph.add_node(TimeDistributedDense(len(glove)), name='tdd', input='seq2seq')
        graph.add_node(Activation('softmax'), name='softmax', input='tdd')

    graph.add_output(name='output', input='softmax')
    loss_fun = hs_categorical_crossentropy if hs else 'categorical_crossentropy'
    graph.compile(loss={'output':loss_fun}, optimizer='adam', sample_weight_modes={'output':'temporal'})
    return graph

def make_adverse_model(generative_model, embed_size = 50, batch_size = 64):
    discriminator = Sequential()
    discriminator.add(LSTM(embed_size, batch_input_shape=(batch_size, HYPO_LEN, embed_size)))
    discriminator.add(Dense(1, activation='sigmoid'))    
    batch_input_shape = (batch_size,PREM_LEN, embed_size)
    
    graph = Graph()
    #graph.add_input(name='premise_input', batch_input_shape=batch_input_shape)
    #graph.add_input(name='embed_input', batch_input_shape=(batch_size,1), dtype='int')
    graph.add_node(generative_model, name = 'generative')#, inputs=['premise_input', 'embed_input'])
    graph.add_input(name='training_hypo', batch_input_shape=(batch_size, HYPO_LEN, embed_size))
    graph.add_shared_node(discriminator, name='shared', inputs=['training_hypo', 'generative'], merge_mode='join')
    
    def margin_opt(inputs):
        print(inputs)
        assert len(inputs) == 2, ('Margin Output needs '
                              '2 inputs, %d given' % len(inputs))
        u, v = inputs.values()
        return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))
    graph.add_node(Lambda(margin_opt), name = 'output2', input='shared', create_output = True)
    graph.cache_enabled = False
    graph.compile(loss={'output2':'mse'}, optimizer='sgd')
    return graph
        
        
def generation_test(train, glove, model, batch_size = 64):
    mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
    p = Progbar(len(train))
    for i, train_index in mb:
        X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
        X_p = load_data.pad_sequences(X_prem, maxlen = PREM_LEN, dim = 50)
        X_h = load_data.pad_sequences(X_hypo, maxlen = HYPO_LEN, dim = 50)
        train_loss =  model.train_on_batch(X_p, X_h)[0]
        p.add(len(X_p),[('train_loss', train_loss)])
        
def generation_embded_test(train, glove, model, batch_size = 64):
    batch = np.arange(batch_size)
    X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in batch], glove)
    X_p = load_data.pad_sequences(X_prem, maxlen = PREM_LEN, dim = 50)
    X_h = load_data.pad_sequences(X_hypo, maxlen = HYPO_LEN, dim = 50)
    data = {'premise_input': X_p, 'embed_input': np.expand_dims(np.array(batch), axis=1), 'output' : X_h}        
    return model.train_on_batch(data)
        #p.add(len(X_p),[('train_loss', train_loss)])
    
def train_model_embed(train, dev, glove, model, nb_epochs = 20, batch_size = 64, worse_steps = 5, hs=True, ci = True):
    X_dev_p, X_dev_h, y_dev = load_data.prepare_split_vec_dataset(dev, glove)
    embed_size = X_dev_p[0].shape[1]
    
    word_index = load_data.WordIndex(glove)

    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], word_index.index)

            padded_p = load_data.pad_sequences(X_train_p, maxlen = PREM_LEN, dim = embed_size, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, maxlen = HYPO_LEN, dim = 1, padding = 'post')
            
            data = {'premise_input': padded_p, 'embed_input': np.expand_dims(np.array(train_index), axis=1), 'output' : padded_h}
            if ci:
               data['class_input'] = y_train
            if hs:
		data['train_input'] = padded_h.sum(axis=2)
                data['output'] = np.ones((batch_size, HYPO_LEN, 1))
            
            sw = (np.sum(padded_h, axis = 2) != 0).astype(float)
            train_loss = float(model.train_on_batch(data, sample_weight={'output':sw})[0])
            p.add(len(train_index),[('train_loss', train_loss)])
        sys.stdout.write('\n')


def generation_predict_embed(model, glove, premise, embed_index, class_index,  batch_size = 64, hs = True, ci = True):
    X_p = load_data.load_word_vecs(premise, glove)
    X_p = load_data.pad_sequences([X_p], maxlen=PREM_LEN, dim = len(X_p[0]))
    X = np.zeros((batch_size, X_p.shape[1], X_p.shape[2]))
    X[0] = X_p[0]
    E = np.zeros((batch_size, 1))
    E[0][0] = embed_index
    
    data = {'premise_input': X, 'embed_input': E}
    
    if ci:
      C = np.zeros((batch_size, 3))
      C[0][class_index] = 1
      data['class_input'] = C
    
    if hs:
        data['train_input'] = np.zeros((batch_size, HYPO_LEN))
    
    model_pred = model.predict_on_batch(data)
    return model_pred['output'][0]

def project(embed_sent, glove):
    result = []
    ind = np.argmax(embed_sent, axis = 1)
    for i in ind:
        result.append(glove.keys()[i])
    return result

def rank_sent(premise, hypothesis, model, glove, glove_mat):
    result = []
    y_pred = generation_predict(model, glove, premise)
    for i in range(len(hypothesis)):
        result.append(get_rank(y_pred[i], hypothesis[i], glove, glove_mat))
    return result    

def get_word(array, glove, glove_mat):
    prod = np.dot(glove_mat, array)
    return glove.keys()[prod.argmax()]

def get_rank(array, word, glove, glove_mat):
    prod = np.dot(glove_mat, array)
    ind = glove.keys().index(word)
    return len(glove) - prod.argsort().argsort()[ind]

def make_glove_mat(glove):
    glove_mat = np.array(glove.values())    
    glove_mat = glove_mat / np.linalg.norm(glove_mat, axis = 1)[:,None]
    return glove_mat

def transform_dataset(dataset, class_str = 'entailment', max_prem_len = sys.maxint, max_hypo_len = sys.maxint):
    uniq = set()
    result = []
    for ex in dataset:
        prem_str = " ".join(ex[0])
        if class_str == None:
            prem_str += ex[2]
        if  (class_str == None or ex[2] == class_str) and prem_str not in uniq and len(ex[0]) <= max_prem_len and len(ex[1]) <= max_hypo_len:
            result.append(ex)
            uniq.add(prem_str)
    return result

def add_eol_to_hypo(dataset):
    for ex in dataset:
        if ex[1][-1] != '.':
            ex[1].append('.')
        
    

def word_overlap(premise, hypo):
    return len([h for h in hypo if h in premise]) / float(len(hypo))    
           

def hypo_to_string(hypo, eos = '.'):
    if eos in hypo:
        ind = hypo.index(eos)
        return " ".join(hypo[:ind + 1])
    else:
        return " ".join(hypo)
 

def test_genmodel(gen_model, train, dev, glove, classify_model = None, glove_alter = None):
    dev_count = 10
    gens_count = 10
   
    for ex in dev[:dev_count]:
        print " ".join(ex[0])
        print " ".join(ex[1])
        print "Generations: "
        for train_index in np.random.random_integers(0, len(train), gens_count):
            pred = generation_predict_embed(gen_model, glove, ex[0], train_index)
            
            gen_lex = project(pred, glove)
            gen_str = hypo_to_string(gen_lex)
            
            if classify_model != None and glove_alter != None: 
                probs = misc.predict_example(" ".join(ex[0]), gen_str, classify_model, glove_alter)
                print probs[0][0][2], gen_str
            else:
                print gen_str
        print
    
if __name__ == "__main__":
    train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
    glove = load_data.import_glove('data/snli_vectors.txt')
    #add_eol_to_hypo(train)
    #add_eol_to_hypo(dev)
    class_input = 'entailment'
    train = transform_dataset(train, class_input, PREM_LEN, HYPO_LEN)
    dev = transform_dataset(dev, class_input, PREM_LEN, HYPO_LEN)
    for ex in train+dev:
        load_data.load_word_vecs(ex[0] + ex[1], glove)
