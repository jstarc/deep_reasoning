# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:29:07 2016

@author: Janez
"""
import sys
sys.path.append('../keras')

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, Recurrent
from keras.models import Graph, Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Merge, Flatten
from keras import activations, initializations
from keras.backend import theano_backend as K

import numpy as np
import generation
import misc
import load_data

from hierarchical_softmax import HierarchicalSoftmax
from hierarchical_softmax import hs_categorical_crossentropy
import theano
from keras.utils.generic_utils import Progbar

def attention_model(hidden_size, embed_size):
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True)
    hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True)    
    attention = LstmAttentionLayer2(hidden_size)
    
    graph = Graph()
    graph.add_input(name='premise_input', input_shape= (None, embed_size))
    graph.add_input(name='hypo_input', input_shape= (None, embed_size))
    graph.add_node(premise_layer, name='premise', input='premise_input')
    graph.add_node(hypo_layer, name='hypo', input='hypo_input')
    graph.add_node(attention, name='attention', inputs=['premise', 'hypo'], merge_mode='join')       
    graph.add_node(Dense(3), name='dense', input='attention')
    graph.add_node(Activation('softmax'), name='softmax', input='dense')
    graph.add_output(name='output', input='softmax')
    graph.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam())
    
    return graph
def seq2seq_model(examples, hidden_size, embed_size, glove, batch_size = 64, prem_len = 22, hypo_len = 13):
   
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True)
    #hypo_layer = LSTM(output_dim= hidden_size, batch_input_shape=(batch_size, 1, embed_size), return_sequences=True, stateful = True)
    hypo_layer = LSTM(output_dim= hidden_size, return_sequences=True)
    attention = LstmAttentionLayer2(hidden_size, return_sequences=True)
    noise_layer = Embedding(examples, embed_size, input_length = 1)
    
  
    
    graph = Graph()
    graph.add_input(name='premise_input', batch_input_shape = (batch_size, prem_len), dtype = 'int32')
    graph.add_node(generation.make_fixed_embeddings(glove, prem_len), name = 'prem_word_vec', input='premise_input')
    graph.add_node(premise_layer, name = 'premise', input='prem_word_vec')
    
    graph.add_input(name='noise_input', batch_input_shape=(batch_size,1), dtype='int32')
    graph.add_node(noise_layer, name='noise_embeddings_pre', input='noise_input')
    graph.add_node(Flatten(), name='noise_embeddings', input='noise_embeddings_pre')
    graph.add_input(name='class_input', batch_input_shape=(batch_size, 3))
    graph.add_node(Dense(hidden_size), inputs=['noise_embeddings', 'class_input'], name ='creative', merge_mode='concat')
    
    graph.add_input(name='hypo_input', batch_input_shape=(batch_size, hypo_len), dtype = 'int32')
    graph.add_node(generation.make_fixed_embeddings(glove, hypo_len), name = 'hypo_word_vec', input='hypo_input')
    graph.add_node(hypo_layer, name = 'hypo', input='hypo_word_vec')
    
    graph.add_node(attention, name='attention', inputs=['premise', 'hypo', 'creative'], merge_mode='join')
    
    graph.add_input(name='train_input', batch_input_shape=(batch_size, hypo_len), dtype='int32')
    graph.add_node(HierarchicalSoftmax(len(glove), input_dim = hidden_size, input_length = hypo_len), 
                   name = 'softmax', inputs=['attention','train_input'], 
                   merge_mode = 'join')
    graph.add_output(name='output', input='softmax')
    
    graph.compile(loss={'output': hs_categorical_crossentropy}, optimizer='adam')
    return graph

def seq2seq_test_model(train_model, examples, hidden_size, embed_size, glove, batch_size = 64, prem_len = 22):
    
    
    graph = Graph()
    
    hypo_layer = LSTM(output_dim= hidden_size, batch_input_shape=(batch_size, 1, embed_size), 
                      return_sequences=True, stateful = True, trainable = False)
    
    
    graph.add_input(name='hypo_input', batch_input_shape=(batch_size, 1), dtype = 'int32')
    graph.add_node(generation.make_fixed_embeddings(glove, 1), name = 'hypo_word_vec', input='hypo_input')
    graph.add_node(hypo_layer, name = 'hypo', input='hypo_word_vec')
    
    graph.add_input(name='premise', batch_input_shape=(batch_size, prem_len, embed_size))
    graph.add_input(name='creative', batch_input_shape=(batch_size, embed_size))
    
    attention = LstmAttentionLayer2(hidden_size, return_sequences=True, stateful = True, trainable = False)
    
    
    graph.add_node(attention, name='attention', inputs=['premise', 'hypo', 'creative'], merge_mode='join')
   
    
    graph.add_input(name='train_input', batch_input_shape=(batch_size, 1), dtype='int32')
    hs = HierarchicalSoftmax(len(glove), input_dim = hidden_size, input_length = 1, trainable = False)
    
    graph.add_node(hs, 
                   name = 'softmax', inputs=['attention','train_input'], 
                   merge_mode = 'join')
    graph.add_output(name='output', input='softmax')
    
    hypo_layer.set_weights(train_model.nodes['hypo'].get_weights())
    attention.set_weights(train_model.nodes['attention'].get_weights())
    hs.set_weights(train_model.nodes['softmax'].get_weights())    
    
    graph.compile(loss={'output': hs_categorical_crossentropy}, optimizer='adam')
    
    func_premise = theano.function([train_model.inputs['premise_input'].get_input()],
                                    train_model.nodes['premise'].get_output(False), 
                                    allow_input_downcast=True)
    func_noise = theano.function([train_model.inputs['noise_input'].get_input(),
                                  train_model.inputs['class_input'].get_input()],
                                  train_model.nodes['creative'].get_output(False),
                                  allow_input_downcast=True)                            
                                    
    return graph, func_premise, func_noise
    
    
    
def train_att_seq2seq(train, dev, glove, model, model_dir = 'models/curr_model', nb_epochs = 20, batch_size = 64, prem_len = 22, hypo_len = 12):
    word_index = load_data.WordIndex(glove)
    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], word_index.index)
            padded_p = load_data.pad_sequences(X_train_p, maxlen = prem_len, dim = -1, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, maxlen = hypo_len, dim = -1, padding = 'post')
            
            hypo_input = np.concatenate([np.zeros((batch_size, 1)), padded_h], axis = 1)
            train_input = np.concatenate([padded_h, np.zeros((batch_size, 1))], axis = 1)
            
            data = {'premise_input': padded_p, 
                    'hypo_input': hypo_input, 
                    'train_input' : train_input,
                    'noise_input' : np.expand_dims(train_index, axis=1),
                    'class_input' : y_train,
                    'output': np.ones((batch_size, hypo_len + 1, 1))}
            
            train_loss = float(model.train_on_batch(data)[0])
            p.add(len(train_index),[('train_loss', train_loss)])
            

def generation_predict_embed(test_model_funcs, word_index, batch, embed_indices, class_indices, batch_size = 64, prem_len = 22, hypo_len = 12):
    prem, _, _ = load_data.prepare_split_vec_dataset(batch, word_index)
    padded_p = load_data.pad_sequences(prem, maxlen=prem_len, dim = -1)
    
    tmodel, premise_func, noise_func = test_model_funcs
    premise = premise_func(padded_p)
    
    noise = noise_func(embed_indices, load_data.convert_to_one_hot(class_indices, 3))
    
    tmodel.reset_states()
    word_input = np.zeros((batch_size, 1))
    result = []
    for i in range(hypo_len):
        data = {'premise' :premise,
                'creative': noise,
                'hypo_input': word_input,
                'train_input': np.zeros((batch_size,1))}
        preds = tmodel.predict_on_batch(data)['output']
        result.append(preds)
    result = np.transpose(np.array(result)[:,:,-1,:], (1,0,2))
    return result
    
    

def test_seq2seq_batch(train, tmodel, premise_func, noise_func, glove, batch_size = 64):
    word_index = load_data.WordIndex(glove)
    X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in range(64)], word_index.index)
    padded_p = load_data.pad_sequences(X_train_p, maxlen = 22, dim = -1, padding = 'pre', dtype='int32')
   
    premise = premise_func(padded_p)
    noise_input = np.expand_dims(np.arange(batch_size), axis=1)
    noise = noise_func(noise_input, y_train)
    
    tmodel.reset_states()
    
    for i in range(12):
        data = {'premise' :premise,
                'creative': noise,
                'hypo_input': padded_h[:,[i]],
                'train_input': np.zeros((64,1))}
        res = tmodel.predict_on_batch(data)
    
    
    
def train_seq2seq_batch(train, model, glove):
    
    word_index = load_data.WordIndex(glove)
    X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in range(64)], word_index.index)
    padded_p = load_data.pad_sequences(X_train_p, maxlen = 22, dim = -1, padding = 'pre')
    padded_h = load_data.pad_sequences(X_train_h, maxlen = 12, dim = -1, padding = 'post')
    
    
    print padded_h.shape
    data = {'premise_input': padded_p, 
            'hypo_input': padded_h, 
            'train_input' : padded_h,
            'noise_input' : np.expand_dims(np.array(range(64)), axis=1),
            'class_input' : y_train,
            'output': np.ones((64, 12, 1))}
    
    train_loss = float(model.train_on_batch(data)[0])
    
def test_genmodel(gen_model, train, dev, word_index, classify_model = None, glove_alter = None, batch_size = 64, ci = False):
   
    gens_count = 2
    dev_counts = 5
    gens = []
    for i in range(gens_count):
        creatives = np.random.random_integers(0, len(train), (batch_size,1))
        preds = generation_predict_embed(gen_model, word_index.index, dev[:batch_size], creatives, class_indices = [i % 3] * batch_size)
        gens.append(generation.get_classes(preds))

    for j in range(dev_counts):
        print " ".join(dev[j][0])
        print " ".join(dev[j][1])
        print "Generations: "
        for i in range(gens_count):
            gen_lex = gens[i][j]
            gen_str = word_index.print_seq(gen_lex)
            if ci:
               gen_str = load_data.LABEL_LIST[i%3][0] + ' ' + gen_str
            
            if classify_model != None and glove_alter != None: 
                probs = misc.predict_example(" ".join(dev[j][0]), gen_str, classify_model, glove_alter)
                print probs[0][0][i%3], gen_str
            else:
                print gen_str
        print
    

    

      
        
class LstmAttentionLayer2(Recurrent):

    def __init__(self, output_dim, init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(LstmAttentionLayer2, self).__init__(**kwargs)

    def set_previous(self, layer):
       self.previous = layer
       self.build()


    @property
    def output_shape(self):
        return (None, self.output_dim)


    def build(self):
        self.W_s = self.init((self.output_dim, self.output_dim))
        self.W_t = self.init((self.output_dim, self.output_dim))
        self.W_a = self.init((self.output_dim, self.output_dim))
        self.w_e = K.zeros((self.output_dim,))

        self.W_i = self.init((self.output_dim * 2, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((self.output_dim * 2, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((self.output_dim * 2, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((self.output_dim * 2, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))
        
        
        self.states = [None, None]
        if self.stateful:
            self.reset_states()
        
        self.params = [self.W_s, self.W_t, self.W_a, self.w_e,
                       self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]
            
    
    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        self.reset = True


    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        self.h_t = X[1]
        
        if not self.stateful or self.reset:
            self.h_s = X[0]
            self.h_init = X[2]

            self.P_j = K.dot(self.h_s, self.W_s)

            self.states = self.get_initial_states(self.h_t)
            self.states[0] = self.h_init
            if self.stateful and self.reset:
                self.reset = False        
        
        last_output, outputs, states = K.rnn(self.step, self.h_t, self.states)
        self.states = states
        return outputs

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.output_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def step(self, x, states):

        P_t = K.dot(x, self.W_t)
        P_a = K.dot(states[0], self.W_a)
        sum3 = self.P_j + P_t.dimshuffle((0,'x',1)) + P_a.dimshuffle((0,'x',1))
        E_kj = K.tanh(sum3).dot(self.w_e)
        Alpha_kj = K.softmax(E_kj)
        weighted = self.h_s * Alpha_kj.dimshuffle((0,1,'x'))
        a_k = weighted.sum(axis = 1)
        m_k = K.T.concatenate([a_k, x], axis = 1)
        
        x_i = K.dot(m_k, self.W_i) + self.b_i
        x_f = K.dot(m_k, self.W_f) + self.b_f
        x_c = K.dot(m_k, self.W_c) + self.b_c
        x_o = K.dot(m_k, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(states[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(states[0], self.U_f))
        c = f * states[1] + i * self.activation(x_c + K.dot(states[0], self.U_c))
        o = self.inner_activation(x_o + K.dot(states[0], self.U_o))
        h = o * self.activation(c)
        
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LstmAttentionLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        
        
        
        
        
