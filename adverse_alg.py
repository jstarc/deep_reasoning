import numpy as np
import os 

import load_data
from generative_alg import generative_predict_beam, make_gen_batch

from keras.callbacks import ModelCheckpoint, EarlyStopping



        

def train_adverse_model(train, dev, adverse_model, generative_model, word_index, model_dir, 
                        nb_epochs, batch_size, hypo_len): 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    train_gen = adverse_generator(train, generative_model, word_index, 0.8, batch_size, 
                                  hypo_len)
    dev_gen = adverse_generator(dev, generative_model, word_index, 0.0, batch_size, hypo_len)
    val_data = prepare_dev_data(dev_gen, len(dev) / batch_size)
    saver = ModelCheckpoint(model_dir + '/model.weights', monitor = 'loss')
    es = EarlyStopping(patience = 5)
    
    return adverse_model.fit_generator(train_gen, samples_per_epoch = 64000, 
                 nb_epoch = nb_epochs, callbacks = [saver, es], validation_data = val_data) 
   
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
    
def adverse_generator(train, gen_model, word_index, cache_prob, batch_size, hypo_len):
    cache =  []    
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             
             orig_batch = [train[k] for k in train_index]
             if np.random.random() > cache_prob or len(cache) < 100:
                 gen_batch, _ = make_gen_batch(orig_batch, gen_model, word_index, hypo_len)
                 cache.append(gen_batch)
             else:
                 gen_batch = cache[np.random.random_integers(0, len(cache) - 1)]
                 
             train_batch = make_train_batch(orig_batch, word_index, hypo_len)
             yield {'train_hypo' : train_batch, 'gen_hypo': gen_batch, 
                    'output2': np.zeros((batch_size))}
        
def make_train_batch(orig_batch, word_index, hypo_len):
    _, X_hypo, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
    return load_data.pad_sequences(X_hypo, maxlen = hypo_len, dim = -1, padding = 'post')
    

def manual_tester(dev, discriminator, generative_model, word_index, batch_size,
                          hypo_len, target_size, filename):
    
    gen = adverse_generator(dev, generative_model, word_index, 0.0, batch_size, hypo_len)
    count = 0
    with open(filename+'.hidden', 'w') as h, open(filename+'.revealed', 'w') as r:
        while count < target_size:
            batch = next(gen)
            gen_preds = discriminator.predict_on_batch(batch['gen_hypo'])[0]
            train_preds = discriminator.predict_on_batch(batch['train_hypo'])[0]
            for i in range(batch_size):
                gen_hypo = word_index.print_seq(batch['gen_hypo'][i])
                train_hypo = word_index.print_seq(batch['train_hypo'][i])
                
                r_data = [gen_hypo, train_hypo, str(gen_preds[i][0]), str(train_preds[i][0])]
                r.write("\t".join(r_data) + '\n')


                h_data = [gen_hypo, train_hypo] if np.random.rand() > 0.5 else [train_hypo, gen_hypo]
                h.write("\t".join(h_data) + '\n')
            count += batch_size
                          
                   
            
