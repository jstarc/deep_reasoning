
import os
import load_data
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping


        
def train_classify_graph(train, dev, wi, model, model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 128):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    g_train = graph_generator(train, batch_size, wi)
    g_dev = graph_generator(dev, batch_size, wi)   
    es = EarlyStopping(patience = 5)
    saver = ModelCheckpoint(model_dir + '/model.weights', monitor = 'val_loss')
    
    return model.fit_generator(g_train, samples_per_epoch = batch_size * 1000, nb_epoch = nb_epochs, 
                               validation_data = g_dev, nb_val_samples = len(dev), show_accuracy=True, 
                               callbacks = [saver, es])
        
        
def graph_generator(train, batch_size, word_index):
    while True:
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        for i, train_index in mb:
            X_train_p, X_train_h, y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], word_index.index)
            padded_p = load_data.pad_sequences(X_train_p, dim = -1, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, dim = -1, padding = 'post')
            yield {'premise_input': padded_p, 'hypo_input': padded_h, 'output' : y_train}

def train_generative_graph(train, wi, model, model_dir =  'models/curr_model', nb_epochs = 20, batch_size = 128):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    g_train = generative_generator(train, wi, batch_size)
    saver = ModelCheckpoint(model_dir + '/model.weights', monitor = 'loss')
    
    return model.fit_generator(g_train, samples_per_epoch = len(train), nb_epoch = nb_epochs,  
                               callbacks = [saver])         
            
def generative_generator(train, word_index, batch_size = 64, prem_len = 22, hypo_len = 12):
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             X_train_p, X_train_h , y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], word_index.index)
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
 

