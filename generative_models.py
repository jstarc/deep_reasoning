import sys
sys.path.append('../seq2seq')

from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector, TimeDistributedDense, Activation, Flatten, Layer

from seq2seq.models import AttentionSeq2seq

from hierarchical_softmax import HierarchicalSoftmax
from hierarchical_softmax import hs_categorical_crossentropy
from common import make_fixed_embeddings
from attention import LstmAttentionLayer

import theano

def create_f_model(examples, glove, hidden_size = 10, embed_size = 50, batch_size = 64, 
                 hs = True, ci = True, prem_len = 22, hypo_len = 12):
    
    batch_input_shape = (batch_size, prem_len, embed_size)
    
    em_model = Sequential()    
    em_model.add(Embedding(examples, embed_size, input_length = 1, batch_input_shape=(batch_size,1)))
    em_model.add(Flatten())
    em_model.add(Dense(embed_size))
    em_model.add(RepeatVector(prem_len))
    
    input_dim = embed_size * 2
    if ci:
        input_dim += 3
    seq2seq = AttentionSeq2seq(
        batch_input_shape = batch_input_shape,
        input_dim = input_dim,
        hidden_dim=embed_size,
        output_dim=embed_size,
        output_length=hypo_len,
        depth=1,
        bidirectional=False,
    )

    class_model = Sequential()
    class_model.add(RepeatVector(prem_len))
    
    graph = Graph()
    graph.add_input(name='premise_input', batch_input_shape=(batch_size, prem_len), dtype = 'int')
    graph.add_node(make_fixed_embeddings(glove, prem_len), name = 'word_vec', input='premise_input')
    
    graph.add_input(name='embed_input', batch_input_shape=(batch_size,1), dtype='int')
    graph.add_node(em_model, name='em_model', input='embed_input')
    
    seq_inputs = ['word_vec', 'em_model']
    
    if ci:
        graph.add_input(name='class_input', batch_input_shape=(batch_size,3))
        graph.add_node(class_model, name='class_model', input='class_input')
        seq_inputs += ['class_model']
   
    graph.add_node(seq2seq, name='seq2seq', inputs=seq_inputs, merge_mode='concat')
    
    if hs: 
        graph.add_input(name='train_input', batch_input_shape=(batch_size, hypo_len), dtype='int32')
        graph.add_node(HierarchicalSoftmax(len(glove), input_dim = embed_size, input_length = hypo_len), 
                   name = 'softmax', inputs=['seq2seq','train_input'], 
                   merge_mode = 'join')
    else:
        graph.add_node(TimeDistributedDense(len(glove)), name='tdd', input='seq2seq')
        graph.add_node(Activation('softmax'), name='softmax', input='tdd')

    graph.add_output(name='output', input='softmax')
    loss_fun = hs_categorical_crossentropy if hs else 'categorical_crossentropy'
    graph.compile(loss={'output':loss_fun}, optimizer='adam', sample_weight_modes={'output':'temporal'})
    return graph
    
def create_o_train_model(examples, hidden_size, embed_size, glove, batch_size = 64, prem_len = 22, hypo_len = 13):
   
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True)
   
    hypo_layer = LSTM(output_dim= hidden_size, return_sequences=True)
    attention = LstmAttentionLayer(hidden_size, return_sequences=True, feed_state = True)
    noise_layer = Embedding(examples, embed_size, input_length = 1)
    

    graph = Graph()
    graph.add_input(name='premise_input', batch_input_shape = (batch_size, prem_len), dtype = 'int32')
    graph.add_node(make_fixed_embeddings(glove, prem_len), name = 'prem_word_vec', input='premise_input')
    graph.add_node(premise_layer, name = 'premise', input='prem_word_vec')
    
    graph.add_input(name='noise_input', batch_input_shape=(batch_size,1), dtype='int32')
    graph.add_node(noise_layer, name='noise_embeddings_pre', input='noise_input')
    graph.add_node(Flatten(), name='noise_embeddings', input='noise_embeddings_pre')
    
    graph.add_input(name='class_input', batch_input_shape=(batch_size, 3))
    graph.add_node(Dense(hidden_size), inputs=['noise_embeddings', 'class_input'], name ='creative', merge_mode='concat')
    
    graph.add_input(name='hypo_input', batch_input_shape=(batch_size, hypo_len), dtype = 'int32')
    graph.add_node(make_fixed_embeddings(glove, hypo_len), name = 'hypo_word_vec', input='hypo_input')
    graph.add_node(hypo_layer, name = 'hypo', input='hypo_word_vec')
    
    graph.add_node(attention, name='attention', inputs=['premise', 'hypo', 'creative'], 
                   merge_mode='join')
    
    graph.add_input(name='train_input', batch_input_shape=(batch_size, hypo_len), dtype='int32')
    graph.add_node(HierarchicalSoftmax(len(glove), input_dim = hidden_size, input_length = hypo_len), 
                   name = 'softmax', inputs=['attention','train_input'], 
                   merge_mode = 'join')
    graph.add_output(name='output', input='softmax')
    
    graph.compile(loss={'output': hs_categorical_crossentropy}, optimizer='adam')
    return graph
    

    
def create_o_test_model(train_model, examples, hidden_size, embed_size, glove, batch_size, prem_len):
    
    
    graph = Graph()
    
    hypo_layer = LSTM(output_dim= hidden_size, batch_input_shape=(batch_size, 1, embed_size), 
                      return_sequences=True, stateful = True, trainable = False)
    
    
    graph.add_input(name='hypo_input', batch_input_shape=(batch_size, 1), dtype = 'int32')
    graph.add_node(make_fixed_embeddings(glove, 1), name = 'hypo_word_vec', input='hypo_input')
    graph.add_node(hypo_layer, name = 'hypo', input='hypo_word_vec')
    
    graph.add_input(name='premise', batch_input_shape=(batch_size, prem_len, embed_size))
    graph.add_input(name='creative', batch_input_shape=(batch_size, embed_size))
    
    attention = LstmAttentionLayer(hidden_size, return_sequences=True, stateful = True, trainable = False, feed_state = False)
    
    
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
    func_noise = theano.function([train_model.nodes['noise_embeddings'].get_input(False),
                                  train_model.inputs['class_input'].get_input()],
                                  train_model.nodes['creative'].get_output(False),
                                  allow_input_downcast=True)                            

    return graph, func_premise, func_noise
    
    
def create_o2_train_model(examples, hidden_size, glove, batch_size, prem_len, hypo_len):
   
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True)
   
    hypo_layer = LSTM(output_dim= hidden_size - 3, return_sequences=True)
    attention = LstmAttentionLayer(hidden_size, return_sequences=True, feed_state = True)
    noise_layer = Embedding(examples, hidden_size, input_length = 1)
    

    graph = Graph()
    graph.add_input(name='premise_input', batch_input_shape = (batch_size, prem_len), dtype = 'int32')
    graph.add_node(make_fixed_embeddings(glove, prem_len), name = 'prem_word_vec', input='premise_input')
    graph.add_node(premise_layer, name = 'premise', input='prem_word_vec')
    
    graph.add_input(name='noise_input', batch_input_shape=(batch_size,1), dtype='int32')
    graph.add_node(noise_layer, name='noise_embeddings_pre', input='noise_input')
    graph.add_node(Flatten(), name='creative', input='noise_embeddings_pre')
    
    graph.add_input(name='class_input', batch_input_shape=(batch_size, 3))
    graph.add_node(RepeatVector(hypo_len + 1), name='class_td', input='class_input')
   
    graph.add_input(name='hypo_input', batch_input_shape=(batch_size, hypo_len + 1), dtype = 'int32')
    graph.add_node(make_fixed_embeddings(glove, hypo_len + 1), name = 'hypo_word_vec', input='hypo_input')
    graph.add_node(hypo_layer, name = 'hypo', input='hypo_word_vec')
    
    graph.add_node(Layer(), inputs=['hypo','class_td'],name ='hypo_merge', 
                   merge_mode = 'concat', concat_axis = 2)
    
    graph.add_node(attention, name='attention', inputs=['premise', 'hypo_merge', 'creative'], 
                   merge_mode='join')
    
    graph.add_input(name='train_input', batch_input_shape=(batch_size, hypo_len + 1), dtype='int32')
    graph.add_node(HierarchicalSoftmax(len(glove), input_dim = hidden_size, input_length = hypo_len + 1), 
                   name = 'softmax', inputs=['attention','train_input'], 
                   merge_mode = 'join')
    graph.add_output(name='output', input='softmax')
    
    graph.compile(loss={'output': hs_categorical_crossentropy}, optimizer='adam')
    return graph
    
    
def create_o2_test_model(train_model, glove):
    
    batch_size, prem_len, hidden_size = train_model.nodes['premise'].output_shape
    embed_size = train_model.nodes['prem_word_vec'].output_shape[2]
    graph = Graph()
    
    hypo_layer = LSTM(output_dim= hidden_size - 3, batch_input_shape=(batch_size, 1, embed_size), 
                      return_sequences=True, stateful = True, trainable = False)
    
    
    graph.add_input(name='hypo_input', batch_input_shape=(batch_size, 1), dtype = 'int32')
    graph.add_node(make_fixed_embeddings(glove, 1), name = 'hypo_word_vec', input='hypo_input')
    graph.add_node(hypo_layer, name = 'hypo', input='hypo_word_vec')
    
    graph.add_input(name='premise', batch_input_shape=(batch_size, prem_len, embed_size))
    graph.add_input(name='creative', batch_input_shape=(batch_size, embed_size))
    
    graph.add_input(name='class_input', batch_input_shape=(batch_size, 3))
    graph.add_node(RepeatVector(1), name='class_td', input='class_input')
    graph.add_node(Layer(), inputs=['hypo','class_td'], name ='hypo_merge', merge_mode = 'concat')
     
    attention = LstmAttentionLayer(hidden_size, return_sequences=True, stateful = True, 
                                   trainable = False, feed_state = False)
    
    
    graph.add_node(attention, name='attention', inputs=['premise', 'hypo_merge', 'creative'], merge_mode='join')
   
    
    graph.add_input(name='train_input', batch_input_shape=(batch_size, 1), dtype='int32')
    hs = HierarchicalSoftmax(len(glove), input_dim = hidden_size, input_length = 1, 
                             trainable = False)
    
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
    func_noise = theano.function([train_model.nodes['creative'].get_input(False)],
                                  train_model.nodes['creative'].get_output(False),
                                  allow_input_downcast=True)                            

    return graph, func_premise, func_noise
    
