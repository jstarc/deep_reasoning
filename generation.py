# -*- coding: utf-8 -*-

import sys
sys.path.append('../seq2seq')
sys.path.append('../keras')
import os
import numpy as np

from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding

from keras.layers.core import Flatten
from keras.layers.core import Dense, RepeatVector, TimeDistributedDense, Activation

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
    
def train_model_embed(train, dev, glove, model, model_dir = 'models/curr_model', nb_epochs = 20, batch_size = 64, worse_steps = 5, hs=True, ci = True):
    X_dev_p, X_dev_h, y_dev = load_data.prepare_split_vec_dataset(dev, glove=glove)
    
    
    word_index = load_data.WordIndex(glove)
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    for e in range(nb_epochs):
        print "Epoch ", e
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        p = Progbar(len(train))
        for i, train_index in mb:
            if len(train_index) != batch_size:
                continue
            X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], word_index.index)
            padded_p = load_data.pad_sequences(X_train_p, maxlen = PREM_LEN, dim = -1, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, maxlen = HYPO_LEN, dim = -1, padding = 'post')
            
            data = {'premise_input': padded_p, 'embed_input': np.expand_dims(np.array(train_index), axis=1), 'output' : padded_h}
            if ci:
                data['class_input'] = y_train
            if hs:
                data['train_input'] = padded_h
                data['output'] = np.ones((batch_size, HYPO_LEN, 1))
            
            #sw = (padded_h != 0).astype(float)
            #train_loss = float(model.train_on_batch(data, sample_weight={'output':sw})[0])
	    train_loss = float(model.train_on_batch(data)[0])
            p.add(len(train_index),[('train_loss', train_loss)])
        sys.stdout.write('\n')
        model.save_weights(model_dir + '/model~' + str(e))


def generation_predict_embed(model, word_index, batch, embed_indices, batch_size = 64, hs = True, class_indices = None):
    prem, hypo, y = load_data.prepare_split_vec_dataset(batch, word_index)
    X_p = load_data.pad_sequences(prem, maxlen=PREM_LEN, dim = -1)
    
    data = {'premise_input': X_p, 'embed_input': embed_indices[:,None]}
    
    if class_indices is not None:
      C = load_data.convert_to_one_hot(class_indices, 3)
      data['class_input'] = C
    
    if hs:
        data['train_input'] = np.zeros((batch_size, HYPO_LEN))
    
    model_pred = model.predict_on_batch(data)
    return model_pred['output']

def get_classes(preds):
   return np.argmax(preds, axis = 2)


def project(embed_sent, word_index):
    result = []
    ind = np.argmax(embed_sent, axis = 1)
    for i in ind:
        result.append(word_index.keys[i])
    return result

 
def get_word(array, glove, glove_mat):
    prod = np.dot(glove_mat, array)
    return glove.keys()[prod.argmax()]


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
 

def test_genmodel(gen_model, train, dev, word_index, classify_model = None, glove_alter = None, batch_size = 64, ci = False):
   
    gens_count = 6
    dev_counts = 6
    gens = []
    for i in range(gens_count):
        creatives = np.random.random_integers(0, len(train), batch_size)
        preds = generation_predict_embed(gen_model, word_index.index, dev[:batch_size], creatives, class_indices = [i % 3] * batch_size)
        gens.append(get_classes(preds))
        
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


if __name__ == "__main__":
    train, dev, test = load_data.load_all_snli_datasets('data/snli_1.0/')
    glove = load_data.import_glove('data/snli_vectors.txt')
    #add_eol_to_hypo(train)
    #add_eol_to_hypo(dev)
    class_input = None
    train = transform_dataset(train, class_input, PREM_LEN, HYPO_LEN)
    dev = transform_dataset(dev, class_input, PREM_LEN, HYPO_LEN)
    for ex in train+dev:
        load_data.load_word_vecs(ex[0] + ex[1], glove)
    load_data.load_word_vec('EOS', glove)
