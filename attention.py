# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:29:07 2016

@author: Janez
"""
import sys
sys.path.append('../keras')

from keras.layers.recurrent import LSTM, Recurrent
from keras.models import Graph
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation
from keras import activations, initializations
from keras.backend import theano_backend as K

import load_data

def attention_model(hidden_size, embed_size):
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True)
    hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True)    
    attention = AttentionLayer(hidden_size)
    
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
    
def graph_train_batch(train, dev, model, glove, embed_size = 300):
    P,H,y = load_data.prepare_split_vec_dataset(train[:154], glove)
    padded_P = load_data.pad_sequences(P, dim = embed_size)
    padded_H = load_data.pad_sequences(H, dim = embed_size)
    data = {'premise_input': padded_P, 'hypo_input': padded_H, 'output' : y}
    return model.train_on_batch(data)
    
    
class AttentionLayer(Recurrent):
    
    def __init__(self, output_dim, init='glorot_uniform', activation='tanh', **kwargs):
        
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def set_previous(self, layer):
       self.previous = layer
       self.build()
        
        
    @property
    def output_shape(self):
        #input_shape = self.previous.layers
        return (None, self.output_dim * 2)
        
                
    def build(self):
        self.W_s = self.init((self.output_dim, self.output_dim))
        self.W_t = self.init((self.output_dim, self.output_dim))
        self.W_a = self.init((self.output_dim, self.output_dim))
        self.V_k = self.init((self.output_dim, self.output_dim))
        self.w_e = K.zeros((self.output_dim,))
        self.states = [None]
        self.params = [self.W_s, self.W_t, self.W_a, self.V_k, self.w_e]
        
        
    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        self.h_s = X[0]
        self.h_t = X[1]
        
        self.P_j = K.dot(self.h_s, self.W_s)
        
        initial_states = self.get_initial_states(self.h_t)

        last_output, outputs, states = K.rnn(self.step, self.h_t, self.output_dim,
                                             initial_states,)
        last_hypo = K.T.unbroadcast(self.h_t.dimshuffle((1,0,2))[-1], 0)
        return K.T.concatenate([last_output, last_hypo], axis = 1)                          
        #return last_output
    
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
        h_ak = a_k + K.tanh(states[0].dot(self.V_k))
        
        return h_ak, [h_ak]
        

        
        
        
        