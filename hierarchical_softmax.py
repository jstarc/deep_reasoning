# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:22:54 2016

@author: Janez
"""
import sys
sys.path.append("../keras/")

from keras.layers.core import Layer, TimeDistributedDense
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Graph
from keras.layers import Activation
from theano.tensor.nnet import h_softmax
import theano.tensor as T
import numpy as np
from keras.utils.generic_utils import Progbar
from numpy.linalg import norm


class HierarchicalSoftmax(Layer): 

    def __init__(self, output_dim, input_dim, input_length, 
                 init='glorot_uniform', **kwargs):
        self.init = initializations.get(init)
        self.output_dim = output_dim
        
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        self.input = K.placeholder(ndim=2)
        
        def hshape(n):
            from math import sqrt, ceil
            l1 = ceil(sqrt(n))
            l2 = ceil(n / l1)
            return int(l1), int(l2)

        self.n_classes, self.n_outputs_per_class = hshape(output_dim)
        super(HierarchicalSoftmax, self).__init__(**kwargs)

    def build(self):
        self.W1 = self.init((self.input_dim, self.n_classes), name='{}_W1'.format(self.name))
        self.b1 = K.zeros((self.n_classes,),  name='{}_b1'.format(self.name))
        self.W2 = self.init((self.n_classes, self.input_dim, self.n_outputs_per_class), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((self.n_classes, self.n_outputs_per_class),  name='{}_b2'.format(self.name))

        self.trainable_weights = [self.W1, self.b1, self.W2, self.b2]


    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        
        x = K.reshape(X[0], (-1, self.input_dim))  # (samples * timesteps, input_dim)
        
        target =  X[1].flatten() if train else None
        
        Y = h_softmax(x, K.shape(x)[0], self.output_dim, 
                              self.n_classes, self.n_outputs_per_class,
                              self.W1, self.b1, self.W2, self.b2, target)
        
        flex_output = 1 if train else self.output_dim
        
        output = K.reshape(Y, (-1, self.input_length, flex_output))
        
        return output
    

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'input_dim': self.input_dim}
        base_config = super(HierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
def hs_categorical_crossentropy(y_true, y_pred):
        return T.nnet.categorical_crossentropy(y_pred, y_true)

def test_hierarchical_softmax(timesteps = 15, input_dim = 50, batch_size = 32,
                              output_dim = 3218, batches = 300, epochs = 30):
    
    model = Graph()
    model.add_input(name='real_input', batch_input_shape=(batch_size, timesteps, input_dim))
    model.add_input(name='train_input', batch_input_shape=(batch_size, timesteps), dtype='int32')
    model.add_node(HierarchicalSoftmax(output_dim, input_dim = input_dim, input_length = timesteps), 
                   name = 'hs', inputs=['real_input','train_input'], 
                   merge_mode = 'join', create_output=True)
    
    
    model.compile(loss={'hs':hs_categorical_crossentropy}, optimizer='adam')
    print "hs model compiled"    
    
    model2 = Sequential()
    model2.add(TimeDistributedDense(output_dim, 
                    batch_input_shape=(batch_size, timesteps, input_dim)))
    model2.add(Activation('softmax'))    
    model2.compile(loss='categorical_crossentropy', optimizer='adam')
    print "softmax model compiled"
    
    learn_f = np.random.normal(size = (input_dim, output_dim))
    learn_f = np.divide(learn_f, norm(learn_f, axis=1)[:,None])
    print "learn_f generated"
    
    
    for j in range(epochs):    
      
      
        batch_data= generate_batch(learn_f, batch_size, 
                                   timesteps, input_dim, output_dim, batches)
            
        print "Epoch", j, "data genrated"
         
        p = Progbar(batches * batch_size)
        for b in batch_data:
            data_train = {'real_input': b[0], 'train_input': b[1], 'hs':b[2]}
            loss =  float(model.train_on_batch(data_train)[0])
            p.add(batch_size,[('hs_loss', loss)])
        p2 = Progbar(batches * batch_size)
        for b in batch_data:
            loss, acc  =  model2.train_on_batch(b[0], b[3], accuracy=True)
            p2.add(batch_size,[('softmax_loss', loss),('softmax_acc', acc)])
           
    
    test_data = generate_batch(learn_f, batch_size, 
                                   timesteps, input_dim, output_dim, batches)
                                   
    p = Progbar(batches * batch_size)
    for b in test_data:
        data_test = {'real_input': b[0], 'train_input': b[1], 'hs':b[3]}
        loss =  float(model.test_on_batch(data_test)[0])
        p.add(batch_size,[('hs__test_loss', loss)])
        
    p2 = Progbar(batches * batch_size)
    for b in batch_data:
        loss =  float(model2.train_on_batch(b[0], b[3])[0])
        p2.add(batch_size,[('softmax_loss', loss)])
        
    
        
def generate_batch(learn_f, batch_size, timesteps, input_dim, output_dim, batches):    
    batch_data = []        
    for i in range(batches):
        ri = np.random.normal(size = (batch_size, timesteps, input_dim))
        ri = np.divide(ri, norm(ri, axis=2)[:,:,None])
        ti = np.argmax(np.dot(ri, learn_f), axis = 2)
       
        ones = np.ones_like(ti, dtype = 'float32')[:,:,None]
        
        one_hot = np.zeros((batch_size, timesteps, output_dim))
        one_hot[np.arange(batch_size)[:,np.newaxis],np.arange(timesteps),ti] = 1.0
        batch_data.append((ri,ti,ones,one_hot))
    return batch_data
    
