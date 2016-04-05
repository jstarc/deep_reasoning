import os
import load_data
import numpy as np
from keras.backend import theano_backend as K 

from keras.callbacks import ModelCheckpoint


def train_generative_graph(train, wi, model, model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 64):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    g_train = generative_train_generator(train, wi, batch_size)
    saver = ModelCheckpoint(model_dir + '/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor = 'loss')
    
    return model.fit_generator(g_train, samples_per_epoch = len(train), nb_epoch = nb_epochs,  
                               callbacks = [saver])         
            

def generative_train_generator(train, word_index, batch_size = 64, prem_len = 22, hypo_len = 12):
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], 
                                                                                  word_index.index)
             padded_p = load_data.pad_sequences(X_train_p, maxlen = prem_len, dim = -1, padding = 'pre')
             padded_h = load_data.pad_sequences(X_train_h, maxlen = hypo_len, dim = -1, padding = 'post')
    
             hypo_input = np.concatenate([np.zeros((batch_size, 1)), padded_h], axis = 1)
             train_input = np.concatenate([padded_h, np.zeros((batch_size, 1))], axis = 1)
             
             yield {'premise_input': padded_p, 
                    'hypo_input': hypo_input, 
                    'train_input' : train_input,
                    'noise_input' : np.expand_dims(train_index, axis=1),
                    'class_input' : y_train,
                    'output': np.ones((batch_size, hypo_len + 1, 1))}
                    
                    
def generative_predict(test_model, word_index, batch, embed_indices, class_indices, batch_size = 64, prem_len = 22, 
                       hypo_len = 12):
    prem, _, _ = load_data.prepare_split_vec_dataset(batch, word_index)
    padded_p = load_data.pad_sequences(prem, maxlen=prem_len, dim = -1)
    
    core_model, premise_func, noise_func = test_model
    premise = premise_func(padded_p)
    
    noise = noise_func(embed_indices, load_data.convert_to_one_hot(class_indices, 3))
    
    core_model.reset_states()
    core_model.nodes['attention'].set_state(noise)

    word_input = np.zeros((batch_size, 1))
    result = []
    for i in range(hypo_len):
        data = {'premise' :premise,
                'creative': noise,
                'hypo_input': word_input,
                'train_input': np.zeros((batch_size,1))}
        preds = core_model.predict_on_batch(data)['output']
        result.append(preds)
        word_input = np.argmax(preds, axis=2)
    result = np.transpose(np.array(result)[:,:,-1,:], (1,0,2))
    return result
    

def generative_predict_beam(test_model, word_index, example, noise_vec, class_index, batch_size = 64, prem_len = 22, 
                       hypo_len = 12):
    prem, _, _ = load_data.prepare_split_vec_dataset([example], word_index.index)
    padded_p = load_data.pad_sequences(prem, maxlen=prem_len, dim = -1)     
    padded_p = np.tile(padded_p[0], (batch_size,1))    
    
    core_model, premise_func, noise_func = test_model
    premise = premise_func(padded_p)
    
    embed_vec = np.tile(noise_vec, (batch_size,1, 1))
    print embed_vec.shape
    noise = noise_func(embed_vec, load_data.convert_to_one_hot(np.repeat(class_index, batch_size), 3))
   
    
    core_model.reset_states()
    core_model.nodes['attention'].set_state(noise)

    word_input = np.zeros((batch_size, 1))
    words = None
    probs = None
    for i in range(hypo_len):
        data = {'premise' :premise,
                'creative': noise,
                'hypo_input': word_input,
                'train_input': np.zeros((batch_size,1))}
        preds = core_model.predict_on_batch(data)['output']
        
        if probs is None:
            word_input = np.argpartition(-preds[0][0], batch_size)[:batch_size]
            probs = preds[0][0][word_input]
            word_input= word_input[:,None]
            words = np.array(word_input)
        else:
            compound_probs =  (preds[:,-1,:] * probs[:, None]).flatten()
            max_indices = np.argpartition(-compound_probs, batch_size)[:batch_size]
            probs = compound_probs[max_indices]
            word_input = (max_indices % preds.shape[-1])[:,None]
            state_indices = max_indices / preds.shape[-1]
            shuffle_states(core_model, state_indices)
            words = np.concatenate([words[state_indices], word_input], axis = -1) 
    return words, probs
    
def shuffle_states(graph_model, indices):
    for l in graph_model.nodes.values():
        if getattr(l, 'stateful', False): 
            for s in l.states:
                K.set_value(s, s[indices].eval())
