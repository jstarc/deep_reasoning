import os
import load_data
import numpy as np

from keras.backend import theano_backend as K 
from keras.callbacks import ModelCheckpoint


def train_generative_graph(train, wi, model, model_dir, batch_size):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    hypo_len = model.get_input_shape_at(0)[1][1] -1
    g_train = generative_train_generator(train, wi, batch_size, hypo_len, 
                                        'control' in model.output_names)
    saver = ModelCheckpoint(model_dir + '/weights.hdf5', monitor = 'loss')
    
    return model.fit_generator(g_train, samples_per_epoch = batch_size * 10000, nb_epoch = 1000,  
                               callbacks = [saver])         
            

def generative_train_generator(train, word_index, batch_size, hypo_len, control):
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], 
                                                                                  word_index.index)
             padded_p = load_data.pad_sequences(X_train_p, dim = -1, padding = 'pre')
             padded_h = load_data.pad_sequences(X_train_h, maxlen = hypo_len, dim = -1, padding = 'post')
    
             hypo_input = np.concatenate([np.zeros((batch_size, 1)), padded_h], axis = 1)
             train_input = np.concatenate([padded_h, np.zeros((batch_size, 1))], axis = 1)
             
             inputs = [padded_p, hypo_input, np.expand_dims(train_index, axis=1), train_input,
                       y_train]
             outputs = [np.ones((batch_size, hypo_len + 1, 1))]
             if control:
                 outputs.append(y_train)
             yield (inputs, outputs)

                    
def generative_predict_beam(test_model, word_index, examples, noise_batch, class_indices, 
                            return_best, hypo_len):
    
    core_model, premise_func, noise_func = test_model
    version = 2 if core_model.get_layer('class_input') else 1
    batch_size = core_model.input_layers[0].input_shape[0]

    beam_size = batch_size / len(examples)
    prem, _, _ = load_data.prepare_split_vec_dataset(examples, word_index.index)
    padded_p = load_data.pad_sequences(prem, dim = -1)     
    padded_p = np.repeat(padded_p, beam_size, axis = 0)    
    premise = premise_func(padded_p)
      
    class_input = load_data.convert_to_one_hot(np.repeat(class_indices, beam_size, axis = 0), 3)
    embed_vec = np.repeat(noise_batch, beam_size, axis = 0)
    if version == 1:
        noise = noise_func(embed_vec, class_input)
    else:
        noise = noise_func(embed_vec)

    core_model.reset_states()
    core_model.get_layer('attention').set_state(noise)

    word_input = np.zeros((batch_size, 1))
    words = None
    probs = None
    for i in range(hypo_len):
        
        data = [premise, word_input, noise, np.zeros((batch_size,1)), class_input]
        if version == 1:
            data = data[:4]
        preds = core_model.predict_on_batch(data)
        split_preds = np.array(np.split(preds, len(examples)))
        if probs is None:
            word_input = np.argpartition(-split_preds[:, 0, 0], beam_size)[:,:beam_size]
            probs = split_preds[:,0,0][np.arange(len(examples))[:, np.newaxis],[word_input]].ravel()
            word_input= word_input.ravel()[:,None]
            words = np.array(word_input)
        else:
            compound_probs =  (preds[:,-1,:] * probs[:, None]).ravel()
            split_cprobs = np.array(np.split(compound_probs, len(examples)))
            max_indices = np.argpartition(-split_cprobs, beam_size)[:,:beam_size]
            probs = split_cprobs[np.arange(len(examples))[:, np.newaxis],[max_indices]].ravel()
            word_input = (max_indices % preds.shape[-1]).ravel()[:,None]
            state_indices = (max_indices / preds.shape[-1]) + np.arange(0, batch_size, beam_size)[:, None]
            state_indices = state_indices.ravel()
            shuffle_states(core_model, state_indices)
            words = np.concatenate([words[state_indices], word_input], axis = -1) 
    if return_best:
        best_ind = np.argmax(np.array(np.split(probs, len(examples))), axis =1) + np.arange(0, batch_size, beam_size)
        return words[best_ind], probs[best_ind]
    else:
        return words, probs
    
def shuffle_states(graph_model, indices):
    for l in graph_model.layers:
        if getattr(l, 'stateful', False): 
            for s in l.states:
                K.set_value(s, s.get_value()[indices])


def make_gen_batch(orig_batch, gen_model, word_index, hypo_len):
    hidden_size = gen_model[0].get_layer('attention').output_shape[2]
    noise_input = np.random.normal(scale=0.11, size=(len(orig_batch), 1, hidden_size))
    class_indices = np.random.random_integers(0, 2, len(orig_batch))

    return generative_predict_beam(gen_model, word_index, orig_batch, noise_input,
                             class_indices, True, hypo_len)
    
