import numpy as np
import os 

import load_data
from generative_alg import generative_predict
from keras.callbacks import ModelCheckpoint, EarlyStopping



        

def train_adverse_model(train, dev, adverse_model, generative_model, word_index, model_dir =  'models/curr_model', 
                        nb_epochs = 20, batch_size = 64, prem_len = 22, hypo_len = 12): 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    train_gen = adverse_generator(train, generative_model, len(train), word_index, batch_size, prem_len, hypo_len)
    dev_gen = adverse_generator(dev, generative_model, len(train), word_index, batch_size, prem_len, hypo_len)
    val_data = prepare_dev_data(dev_gen, len(dev)/ batch_size)
    saver = ModelCheckpoint(model_dir + '/model.weights', monitor = 'loss')
    es = EarlyStopping(patience = 5)
    
    return adverse_model.fit_generator(train_gen, samples_per_epoch = batch_size * 1000, nb_epoch = nb_epochs,  
                               callbacks = [saver, es], 
                               validation_data = val_data) 
    
def prepare_dev_data(dev_gen, batches):
    dev_dicts = [next(dev_gen) for _ in range(batches)]
    merge_dict = {}
    for d in dev_dicts:
        for k, v in d.iteritems():
            merge_dict.setdefault(k, []).append(v)
    result = {}
    for k, v in merge_dict.iteritems():
       result[k] = np.concatenate(v)
    return result
    
def adverse_generator(train, gen_model, noise_embed_len, word_index, batch_size = 64, prem_len = 22, hypo_len = 12):
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             
             orig_batch = [train[k] for k in train_index]
             gen_batch = make_gen_batch(orig_batch, gen_model, noise_embed_len, word_index, batch_size, prem_len, hypo_len)
             train_batch = make_train_batch(orig_batch, word_index, hypo_len)
             yield {'train_hypo' : train_batch, 'gen_hypo': gen_batch, 'output2': np.zeros((batch_size))}
        
def make_gen_batch(orig_batch, gen_model, noise_embed_len, word_index, batch_size = 64, prem_len = 22, hypo_len = 12):
    noise_input = np.random.random_integers(0, noise_embed_len -1, (len(orig_batch), 1))
    class_indices = np.random.random_integers(0, 2, len(orig_batch))
    
    probs = generative_predict(gen_model, word_index.index, orig_batch, noise_input, class_indices, batch_size,
                               prem_len, hypo_len)
    return np.argmax(probs, axis = 2)
    
def make_train_batch(orig_batch, word_index, hypo_len = 12):
    _, X_hypo, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
    return load_data.pad_sequences(X_hypo, maxlen = hypo_len, dim = -1, padding = 'post')
    

