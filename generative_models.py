from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras import backend as K

from hierarchical_softmax import HierarchicalSoftmax
from hierarchical_softmax import hs_categorical_crossentropy
from common import make_fixed_embeddings
from attention import LstmAttentionLayer

import theano


    
def gen_train(noise_examples, hidden_size, glove, hypo_len, version = 1, 
                 control_layer = True):
    prem_input = Input(shape=(None,), dtype='int32', name='prem_input')
    hypo_input = Input(shape=(hypo_len + 1,), dtype='int32', name='hypo_input')
    noise_input = Input(shape=(1,), dtype='int32', name='noise_input')
    train_input = Input(shape=(None,), dtype='int32', name='train_input')
    class_input = Input(shape=(3,), name='class_input')
    
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, hypo_len + 1)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid', name='premise')(prem_embeddings)
    
    if version == 1 or version == 3:
        hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    elif version == 2:
        pre_hypo_layer = LSTM(output_dim=hidden_size - 3, return_sequences=True, 
                            inner_activation='sigmoid', name='hypo')(hypo_embeddings)
        class_repeat = RepeatVector(hypo_len + 1)(class_input)
        hypo_layer = merge([pre_hypo_layer, class_repeat], mode='concat')
    
    noise_layer = Embedding(noise_examples, hidden_size, 
                            input_length = 1, name='noise_embeddings')(noise_input)
    flat_noise = Flatten(name='noise_flatten')(noise_layer)
    if version == 1:
        creative = Dense(hidden_size, activation='tanh', name='creative')\
                        (merge([class_input, flat_noise], mode='concat'))
    elif version == 2 or version == 3:
        creative = flat_noise
            
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True, 
                    feed_state = True, name='attention') ([premise_layer, hypo_layer, creative])
               
    hs = HierarchicalSoftmax(len(glove), trainable = True, name='hs')([attention, train_input])
    
    if control_layer: 
        control_lstm = LSTM(hidden_size, inner_activation='sigmoid')(attention)
        control = Dense(3, activation='softmax', name='control')(control_lstm)
    
    inputs = [prem_input, hypo_input, noise_input, train_input, class_input]
    if version == 3:
        inputs = inputs[:4]
    outputs = [hs, control] if control_layer else [hs]         
    
    model = Model(input=inputs, output=outputs)
    if control_layer:                                                          
        model.compile(loss=[hs_categorical_crossentropy, 'categorical_crossentropy'],  
                      optimizer='adam', loss_weights = [1.0, 0.01],
                      metrics={'hs':word_loss, 'control':[cc_loss, 'acc']})
    else:                                                                              
        model.compile(loss=hs_categorical_crossentropy, optimizer='adam')              
    
    return model
    
def gen_test(train_model, glove, batch_size):
    
    version = 1 if train_model.get_layer('creative') else 2    
    hidden_size = train_model.get_layer('premise').output_shape[-1] 
    
    premise_input = Input(batch_shape=(batch_size, None, None))
    hypo_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    creative_input = Input(batch_shape=(batch_size, None))
    train_input = Input(batch_shape=(batch_size, 1), dtype='int32')
    
    hypo_embeddings = make_fixed_embeddings(glove, 1)(hypo_input) 
    
    if version == 1:
        hypo_layer = LSTM(output_dim = hidden_size, return_sequences=True, stateful = True, 
            trainable = False, inner_activation='sigmoid', name='hypo')(hypo_embeddings)
    elif version == 2:
        pre_hypo_layer = LSTM(output_dim=hidden_size - 3, return_sequences=True, stateful = True, 
            trainable = False, inner_activation='sigmoid', name='hypo')(hypo_embeddings)
        class_input = Input(batch_shape=(64, 3,), name='class_input')
        class_repeat = RepeatVector(1)(class_input)
        hypo_layer = merge([pre_hypo_layer, class_repeat], mode='concat')     
    
    attention = LstmAttentionLayer(output_dim=hidden_size, return_sequences=True, stateful = True,
        trainable = False, feed_state = False, name='attention') \
            ([premise_input, hypo_layer, creative_input])

    hs = HierarchicalSoftmax(len(glove), trainable = False, name ='hs')([attention, train_input])
    
    
    inputs = [premise_input, hypo_input, creative_input, train_input]
    if version == 2:
        inputs.append(class_input)
    outputs = [hs]    
         
    model = Model(input=inputs, output=outputs)
    model.compile(loss=hs_categorical_crossentropy, optimizer='adam')
    
    update_gen_weights(model, train_model)
    
    func_premise = theano.function([train_model.get_layer('prem_input').input],
                                    train_model.get_layer('premise').output, 
                                    allow_input_downcast=True)
    if version == 1:   
        f_inputs = [train_model.get_layer('noise_embeddings').output,
                    train_model.get_layer('class_input').input]
        func_noise = theano.function(f_inputs, train_model.get_layer('creative').output, 
                                     allow_input_downcast=True)                            
    elif version == 2:
        noise = train_model.get_layer('noise_flatten')
        func_noise = theano.function([noise.get_input_at(0)], noise.output, 
                                      allow_input_downcast=True) 
    return model, func_premise, func_noise

def update_gen_weights(test_model, train_model):
    test_model.get_layer('hypo').set_weights(train_model.get_layer('hypo').get_weights())
    test_model.get_layer('attention').set_weights(train_model.get_layer('attention').get_weights())
    test_model.get_layer('hs').set_weights(train_model.get_layer('hs').get_weights()) 
    
def word_loss(y_true, y_pred):
    return K.mean(hs_categorical_crossentropy(y_true, y_pred))
def cc_loss(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, y_true))
