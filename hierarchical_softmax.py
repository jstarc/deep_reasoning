# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:22:54 2016

@author: Janez
"""
import sys
sys.path.append("../keras/")

from keras.layers.core import Layer
from keras import backend as K
from keras import initializations

from theano.tensor.nnet import h_softmax
import theano.tensor as T

from keras.backend.common import _EPSILON 	

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
        return (self.input_length, self.output_dim)

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
                  'input_dim': self.input_dim,
                  'input_length' : self.input_length}
        base_config = super(HierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
def hs_categorical_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)    
    return T.nnet.categorical_crossentropy(y_pred, y_true)


    
