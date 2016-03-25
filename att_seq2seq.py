# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:29:07 2016

@author: Janez
"""
import sys
sys.path.append('../keras')



import numpy as np
import load_data

import theano
from keras.utils.generic_utils import Progbar


        

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
    tmodel.nodes['attention'].feed_state(noise)

    word_input = np.zeros((batch_size, 1))
    result = []
    for i in range(hypo_len):
        data = {'premise' :premise,
                'creative': noise,
                'hypo_input': word_input,
                'train_input': np.zeros((batch_size,1))}
        preds = tmodel.predict_on_batch(data)['output']
        result.append(preds)
        word_input = np.argmax(preds, axis=2)
    result = np.transpose(np.array(result)[:,:,-1,:], (1,0,2))
    return result

def generation_predict_embed_beam(test_model_funcs, word_index, example, embed_index, class_index, batch_size = 64, prem_len = 22, hypo_len = 12):
    prem, _, _ = load_data.prepare_split_vec_dataset([example], word_index)
    padded_p = load_data.pad_sequences(prem, maxlen=prem_len, dim = -1)
    padded_p = np.tile(padded_p, (batch_size, 1))

    tmodel, premise_func, noise_func = test_model_funcs
    premise = premise_func(padded_p)

    embed_indices = np.tile(embed_index, batch_size)
    class_indices = np.tile(class_index, batch_size)
    noise = noise_func(embed_indices, load_data.convert_to_one_hot(class_indices, 3))

    tmodel.reset_states()
    tmodel.nodes['attention'].feed_state(noise)

    word_input = np.zeros((batch_size, 1))
    result = []
    for i in range(hypo_len):
        data = {'premise' :premise,
                'creative': noise,
                'hypo_input': word_input,
                'train_input': np.zeros((batch_size,1))}
        preds = tmodel.predict_on_batch(data)['output']
        result.append(preds)
        word_input = np.argmax(preds, axis=2)
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
    from generative_alg import generative_predict
    gens_count = 10
    dev_counts = 10
    c_vec = []
    class_i  = np.array([load_data.LABEL_LIST.index(ex[2]) for ex in dev[:batch_size]])
    gens = []
    for i in range(gens_count):
        creatives = np.random.random_integers(0, len(train), (batch_size,1))
        preds = generative_predict(gen_model, word_index.index, dev[:batch_size], creatives, 
               class_indices = class_i)
        gens.append(np.argmax(preds, axis = 2))
        c_vec.append(creatives)

    for j in range(dev_counts):
        print " ".join(dev[j][0])
        print dev[j][2]
        
        print "Generations: "
        for i in range(gens_count):
            gen_lex = gens[i][j]
            gen_str = word_index.print_seq(gen_lex)
            #if ci:
            #   gen_str = load_data.LABEL_LIST[i%3][0] + ' ' + gen_str
            #if classify_model != None and glove_alter != None: 
            #    probs = misc.predict_example(" ".join(dev[j][0]), gen_str, classify_model, glove_alter)
            #    print probs[0][0][i%3], gen_str
            #else:
            #    print gen_str
            print gen_str
        print
    

def debug_models(model, tmodel, train, wi):
    X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in range(64)], wi.index)
    padded_p = load_data.pad_sequences(X_train_p, maxlen = 22, dim = -1, padding = 'pre')
    padded_h = load_data.pad_sequences(X_train_h, maxlen = 12, dim = -1, padding = 'post')
    hypo_input = np.concatenate([np.zeros((64, 1)), padded_h], axis = 1)
    train_input = np.concatenate([padded_h, np.zeros((64, 1))], axis = 1)
    noise_input = np.arange(64)[:,None]    
    pi = model.inputs['premise_input'].get_input()
    hi = model.inputs['hypo_input'].get_input()
    ti = model.inputs['train_input'].get_input()
    ni = model.inputs['noise_input'].get_input()
    ci = model.inputs['class_input'].get_input()
    out = model.outputs['output'].get_output(True)
    cout = model.nodes['creative'].get_output(True)
    pout = model.nodes['premise'].get_output(True)
    hout = model.nodes['hypo'].get_output(True)
    aout = model.nodes['attention'].get_output(True)
    ff = theano.function([pi,hi,ti,ni,ci], [out, cout, pout,hout, aout], allow_input_downcast=True)
    
    word_input =  np.zeros((64, 1))
    res5 = ff(padded_p, hypo_input, train_input, noise_input, y_train)
    
    tmhi = tmodel[0].inputs['hypo_input'].get_input()
    tmp = tmodel[0].inputs['premise'].get_input()
    tmc = tmodel[0].inputs['creative'].get_input()
    tmt = tmodel[0].inputs['train_input'].get_input()     
    premise = tmodel[1](padded_p)
    noise = tmodel[2](noise_input, y_train)
   
    print (premise == res5[2]).all()
    print (noise == res5[1]).all()
    
    
    #hypo_f = theano.function([tmhi], tmodel[0].nodes['hypo'].get_output(False), allow_input_downcast = True,
    #                             updates=tmodel[0].nodes['hypo'].updates)
    
    #ho = hypo_f(word_input)
    #ho2 = hypo_f(word_input)
    #print ho[0][0] == ho2[0][0]
    #print (ho[0][0] == res5[3][0][0]).all()
    #print type(tmodel[0].nodes['attention'].states[0])
    tmodel[0].reset_states()
    tmodel[0].nodes['attention'].feed_state(noise)
    #print (tmodel[0].nodes['attention'].states[0].get_value() == noise).all() 
    #taout = tmodel[0].nodes['attention'].get_output(False)
    #att_test_fun =  theano.function([tmhi, tmp, tmc], taout, allow_input_downcast = True, on_unused_input = 'ignore')
    
    #atto = att_test_fun(word_input, premise, noise)

    #print (atto[0][0] == res5[4][0][0]).all()
   
    att_test_fun2 =  theano.function([tmhi, tmp, tmc, tmt], tmodel[0].outputs['output'].get_output(False), 
                         allow_input_downcast = True, on_unused_input = 'ignore', updates=tmodel[0].state_updates)
    atto2 = att_test_fun2(word_input, premise, noise, word_input)
    
    print (res5[0][0][0] == atto2[0][0][padded_h[0][0]]).all()
    print np.argmax(atto2[0][0]), padded_h[0][0]

    atto3 = att_test_fun2(padded_h[:,0, None], premise, noise, word_input)
    print (res5[0][0][1] == atto3[0][0][padded_h[0][1]]).all()
    print res5[0][0][1], atto3[0][0][padded_h[0][1]]
    print np.argmax(atto3[0][0]), padded_h[0][1]
    return res5, atto2, atto3
