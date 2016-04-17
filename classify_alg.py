import os
import load_data

from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_classify_graph(train, dev, wi, model, model_dir, nb_epochs, batch_size):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    g_train = graph_generator(train, batch_size, wi)
    g_dev = graph_generator(dev, batch_size, wi)   
    es = EarlyStopping(patience = 8)
    saver = ModelCheckpoint(model_dir + '/model.weights', monitor = 'val_loss')
    
    return model.fit_generator(g_train, samples_per_epoch = batch_size * 1000, nb_epoch = nb_epochs, 
                               validation_data = g_dev, nb_val_samples = len(dev), 
                               callbacks = [saver, es])
        
        
def graph_generator(train, batch_size, word_index):
    while True:
        mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
        for i, train_index in mb:
            X_train_p, X_train_h, y_train = load_data.prepare_split_vec_dataset([train[k] for k in train_index], 
                                                   word_index.index)
            padded_p = load_data.pad_sequences(X_train_p, dim = -1, padding = 'pre')
            padded_h = load_data.pad_sequences(X_train_h, dim = -1, padding = 'post')
            yield ([padded_p, padded_h], y_train)
