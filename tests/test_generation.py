
import load_data
from keras.utils.generic_utils import Progbar
import numpy as np

def generation_test(train, glove, model, batch_size = 64, prem_len = 22, hypo_len = 12):
    mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)
    p = Progbar(len(train))
    for i, train_index in mb:
        X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in train_index], glove)
        X_p = load_data.pad_sequences(X_prem, maxlen = prem_len, dim = 50)
        X_h = load_data.pad_sequences(X_hypo, maxlen = hypo_len, dim = 50)
        train_loss =  model.train_on_batch(X_p, X_h)[0]
        p.add(len(X_p),[('train_loss', train_loss)])
        
def generation_embded_test(train, glove, model, batch_size = 64, prem_len = 22, hypo_len = 12):
    batch = np.arange(batch_size)
    X_prem, X_hypo, _ = load_data.prepare_split_vec_dataset([train[k] for k in batch], glove)
    X_p = load_data.pad_sequences(X_prem, maxlen = prem_len, dim = 50)
    X_h = load_data.pad_sequences(X_hypo, maxlen = hypo_len, dim = 50)
    data = {'premise_input': X_p, 'embed_input': np.expand_dims(np.array(batch), axis=1), 'output' : X_h}        
    return model.train_on_batch(data)
        #p.add(len(X_p),[('train_loss', train_loss)])