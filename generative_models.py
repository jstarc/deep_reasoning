from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector, Flatten, Lambda
from keras.layers import Input, merge
from keras.models import Model
from keras import backend as K

from hierarchical_softmax import HierarchicalSoftmax
from hierarchical_softmax import hs_categorical_crossentropy
from common import make_fixed_embeddings
from attention import LstmAttentionLayer, FeedLSTM

import theano
import numpy as np

    
def gen_train(noise_examples, hidden_size, noise_dim, glove, hypo_len, version):
    
    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    noise_input = Input(shape=(1,), dtype='int32', name='noise_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')
    
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid', name='premise')(prem_embeddings)
    
    if version == 1 or version == 3 or version == 4 or version == 8:
        hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    elif version == 0 or version == 2 or version == 5:
        pre_hypo_layer = LSTM(output_dim=hidden_size - 3, return_sequences=True, 
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
        class_repeat = RepeatVector(hypo_len + 1)(class_input)
        hypo_layer = merge([pre_hypo_layer, class_repeat], mode='concat')
    
    if version == 0:
        attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True,
                        feed_state = False, name='attention') ([hypo_layer, premise_layer])
    else:
        noise_layer = Embedding(noise_examples, noise_dim, 
                            input_length = 1, name='noise_embeddings')(noise_input)
        flat_noise = Flatten(name='noise_flatten')(noise_layer)
        if version == 1:
            creative = merge([class_input, flat_noise], mode='concat', name = 'cmerge')
        elif version == 2 or version == 3:
            creative = flat_noise
        elif version == 4:
            class_sig = Dense(noise_dim, name = 'class_sig')(class_input)
            creative = merge([flat_noise, class_sig], mode = 'mul', name='cmerge')
        elif version == 5:
            creative = Dense(hidden_size, name = 'class_exp')(flat_noise)
        elif version == 8:
            merged = merge([class_input, flat_noise], mode='concat')
            creative = Dense(hidden_size, name = 'cmerge')(merged)

        attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True, 
                        feed_state = True, name='attention') ([hypo_layer, premise_layer, creative])
               
    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([attention, train_input])
    
    inputs = [prem_input, hypo_input, noise_input, train_input, class_input]
    if version == 3:
        inputs = inputs[:4]
    elif version == 0:
        inputs = inputs[0:2] + inputs[3:5]
    
    model_name = 'version' + str(version)
    model = Model(input=inputs, output=hs, name = model_name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')              
    
    return model

def baseline_train(noise_examples, hidden_size, noise_dim, glove, hypo_len, version):
    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    noise_input = Input(shape=(1,), dtype='int32', name='noise_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')
   
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)

    premise_layer = LSTM(output_dim=hidden_size, return_sequences=False,
                            inner_activation='sigmoid', name='premise')(prem_embeddings)
   
    hypo_layer = FeedLSTM(output_dim=hidden_size, return_sequences=True,
                         feed_layer = premise_layer, name='hypo')([hypo_embeddings])
    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([hypo_layer, train_input])
    inputs = [prem_input, hypo_input, noise_input, train_input, class_input]


    model_name = 'version' + str(version)
    model = Model(input=inputs, output=hs, name = model_name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')

    return model
    
    
def gen_test(train_model, glove, batch_size):
    
    version = int(train_model.name[-1])

    hidden_size = train_model.get_layer('premise').output_shape[-1] 
    
    premise_input = Input(batch_shape=(batch_size, None, None))
    hypo_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    creative_input = Input(batch_shape=(batch_size, None))
    train_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    
    hypo_embeddings = make_fixed_embeddings(glove, 1)(hypo_input) 
    
    if version == 1 or version == 3 or version == 4 or version == 8:
        hypo_layer = LSTM(output_dim = hidden_size, return_sequences=True, stateful = True, unroll=True,
            trainable = False, inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    elif version == 0 or version == 2 or version == 5 or version == 6 or version == 7:
        pre_hypo_layer = LSTM(output_dim=hidden_size - 3, return_sequences=True, stateful = True, 
            trainable = False, inner_activation='sigmoid', name='hypo')(hypo_embeddings)
        class_input = Input(batch_shape=(64, 3,), name='class_input')
        class_repeat = RepeatVector(1)(class_input)
        hypo_layer = merge([pre_hypo_layer, class_repeat], mode='concat')     
    
    att_inputs = [hypo_layer, premise_input] if version == 0 else [hypo_layer, premise_input, creative_input] 
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True, stateful = True, unroll =True,
        trainable = False, feed_state = False, name='attention') \
            (att_inputs)

    hs = HierarchicalSoftmax(len(glove), trainable = False, name ='hs')([attention, train_input])
    
    inputs = [premise_input, hypo_input] + ([] if version == 0 else [creative_input]) + [train_input]
    if version == 0 or version == 2 or version == 5 or version == 6 or version == 7:
        inputs.append(class_input)
    outputs = [hs]    
         
    model = Model(input=inputs, output=outputs, name=train_model.name)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')
    
    update_gen_weights(model, train_model)
    
    func_premise = theano.function([train_model.get_layer('prem_input').input],
                                    train_model.get_layer('premise').output, 
                                    allow_input_downcast=True)
    if version == 1 or version == 4 or version == 8:   
        f_inputs = [train_model.get_layer('noise_embeddings').output,
                    train_model.get_layer('class_input').input]
        func_noise = theano.function(f_inputs, train_model.get_layer('cmerge').output, 
                                     allow_input_downcast=True)                            
    elif version == 2 or version == 3 or version == 5 or version >= 6:
        if version >= 6:
           noise_input = train_model.get_layer('expansion').get_input_at(0)
        else:
           noise_input = train_model.get_layer('noise_flatten').get_input_at(0)
        noise_output = train_model.get_layer('attention').get_input_at(0)[2]
         
        func_noise = theano.function([noise_input], noise_output, 
                                      allow_input_downcast=True) 
              
    elif version == 0:
        func_noise = None
    return model, func_premise, func_noise

def update_gen_weights(test_model, train_model):
    test_model.get_layer('hypo').set_weights(train_model.get_layer('hypo').get_weights())
    test_model.get_layer('attention').set_weights(train_model.get_layer('attention').get_weights())
    test_model.get_layer('hs').set_weights(train_model.get_layer('hs').get_weights()) 
    
def word_loss(y_true, y_pred):
    return K.mean(hs_categorical_crossentropy(y_true, y_pred))
def cc_loss(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, y_true))





def autoe_train(hidden_size, noise_dim, glove, hypo_len, version):

    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')

    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True,
                            inner_activation='sigmoid', name='premise')(prem_embeddings)

    pre_hypo_layer = LSTM(output_dim=hidden_size - 3, return_sequences=True,
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    class_repeat = RepeatVector(hypo_len + 1)(class_input)
    hypo_layer = merge([pre_hypo_layer, class_repeat], mode='concat')

    encoder = LstmAttentionLayer(output_dim=hidden_size, return_sequences=False,
                  feed_state = False, name='encoder') ([hypo_layer, premise_layer])
    if version == 6:
        reduction = Dense(noise_dim, name='reduction', activation='tanh')(encoder)
    elif version == 7:
        z_mean = Dense(noise_dim, name='z_mean')(encoder)
        z_log_sigma = Dense(noise_dim, name='z_log_sigma')(encoder)
          
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(64, noise_dim,),
                              mean=0., std=0.01)
            return z_mean + K.exp(z_log_sigma) * epsilon
        reduction = Lambda(sampling, output_shape=lambda sh: (sh[0][0], noise_dim,), name = 'reduction')([z_mean, z_log_sigma])
        def vae_loss(args):
            z_mean, z_log_sigma = args
            return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)    
        vae = Lambda(vae_loss, output_shape=lambda sh: (sh[0][0], 1,), name = 'vae_output')([z_mean, z_log_sigma])

    creative = Dense(hidden_size, name = 'expansion', activation ='tanh')(reduction)
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True,
                     feed_state = True, name='attention') ([hypo_layer, premise_layer, creative])

    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([attention, train_input])

    inputs = [prem_input, hypo_input, train_input, class_input]

    model_name = 'version' + str(version)
    model = Model(input=inputs, output=(hs if version == 6 else [hs, vae]), name = model_name)
    if version == 6:
        model.compile(loss=hs_categorical_crossentropy, optimizer='adam')
    elif version == 7:
        def minimize(y_true, y_pred):
            return y_pred
        def metric(y_true, y_pred):
            return K.mean(y_pred)
        model.compile(loss=[hs_categorical_crossentropy, minimize], metrics={'hs':word_loss, 'vae_output': metric}, optimizer='adam')
    return model


