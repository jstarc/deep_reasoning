import os
import load_data
import numpy as np

from keras.backend import theano_backend as K 
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import Progbar
from keras.callbacks import Callback
import generative_models as gm


def train(train, dev, model, model_dir, batch_size, glove, beam_size, cmodel = None):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    hypo_len = model.get_input_shape_at(0)[1][1] -1
    g_train = generative_train_generator(train, batch_size, hypo_len, 
                                        'control' in model.output_names)
    saver = ModelCheckpoint(model_dir + '/weights.hdf5', monitor = 'loss')
    
    gtest = gm.gen_test(model, glove, batch_size)
    cb = ValidateGen(dev, gtest, beam_size, hypo_len, 2 ** 10, cmodel)
    
    hist = model.fit_generator(g_train, samples_per_epoch = 2 ** 17, nb_epoch = nb_epochs,  
                               callbacks = [saver, cb])
    return hist
            

def generative_train_generator(train, batch_size, hypo_len, control):
    while True:
         mb = load_data.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
        
         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue
             padded_p = train[0][train_index]
             padded_h = train[1][train_index]
             y_train  = train[2][train_index]
             hypo_input = np.concatenate([np.zeros((batch_size, 1)), padded_h], axis = 1)
             train_input = np.concatenate([padded_h, np.zeros((batch_size, 1))], axis = 1)
             
             inputs = [padded_p, hypo_input, train_index[:, None], train_input,
                       y_train]
             outputs = [np.ones((batch_size, hypo_len + 1, 1))]
             if control:
                 outputs.append(y_train)
             yield (inputs, outputs)

                    
def generative_predict_beam(test_model, premises, noise_batch, class_indices, return_best, 
                            hypo_len):
    
    core_model, premise_func, noise_func = test_model
    version = 2 if core_model.get_layer('class_input') else 1
    batch_size = core_model.input_layers[0].input_shape[0]
    
    beam_size = batch_size / len(premises)
    dup_premises = np.repeat(premises, beam_size, axis = 0)
    premise = premise_func(dup_premises)
      
    class_input = np.repeat(class_indices, beam_size, axis = 0)
    embed_vec = np.repeat(noise_batch, beam_size, axis = 0)
    if version == 1:
        noise = noise_func(embed_vec, class_input)
    else:
        noise = noise_func(embed_vec)

    core_model.reset_states()
    core_model.get_layer('attention').set_state(noise)

    word_input = np.zeros((batch_size, 1))
    result_probs = np.zeros(batch_size)
    lengths = np.zeros(batch_size)
    words = None
    probs = None
    for i in range(hypo_len):
        
        data = [premise, word_input, noise, np.zeros((batch_size,1)), class_input]
        if version == 1 or version == 3:
            data = data[:4]
        preds = core_model.predict_on_batch(data)
        preds = np.log(preds)
        split_preds = np.array(np.split(preds, len(premises)))
        if probs is None:
            word_input = np.argpartition(-split_preds[:, 0, 0], beam_size)[:,:beam_size]
            probs = split_preds[:,0,0][np.arange(len(premises))[:, np.newaxis],[word_input]].ravel()
            word_input= word_input.ravel()[:,None]
            words = np.array(word_input)
        else:
            compound_probs =  (preds[:,-1,:] + probs[:, None]).ravel()
            split_cprobs = np.array(np.split(compound_probs, len(premises)))
            max_indices = np.argpartition(-split_cprobs, beam_size)[:,:beam_size]
            probs = split_cprobs[np.arange(len(premises))[:, np.newaxis],[max_indices]].ravel()
            word_input = (max_indices % preds.shape[-1]).ravel()[:,None]
            state_indices = (max_indices / preds.shape[-1]) + np.arange(0, batch_size, beam_size)[:, None]
            state_indices = state_indices.ravel()
            shuffle_states(core_model, state_indices)
            words = np.concatenate([words[state_indices], word_input], axis = -1)
        result_probs += probs * (word_input[:,0] > 0).astype('int')
        lengths += 1 * (word_input[:,0] > 0).astype('int')
        if (np.sum(word_input) == 0):
            words = np.concatenate([words, np.zeros((batch_size, hypo_len - words.shape[1]))], 
                                    axis = -1)
            break
    
    result_probs /= -lengths
    if return_best:
        best_ind = np.argmax(np.array(np.split(probs, len(premises))), axis =1) + np.arange(0, batch_size, beam_size)
        return words[best_ind], result_probs[best_ind]
    else:
        return words, result_probs
    
def shuffle_states(graph_model, indices):
    for l in graph_model.layers:
        if getattr(l, 'stateful', False): 
            for s in l.states:
                K.set_value(s, s.get_value()[indices])
                
                
def val_generator(dev, gen_test, beam_size, hypo_len):
    batch_size = gen_test[0].input_layers[0].input_shape[0]
    hidden_size = gen_test[0].get_layer('attention').output_shape[2]
    
    per_batch  = batch_size / beam_size
    while True:
        mb = load_data.get_minibatches_idx(len(dev[0]), per_batch, shuffle=True)
        for i, train_index in mb:
            if len(train_index) != per_batch:
               continue
            premises = dev[0][train_index]
            noise_input = np.random.normal(scale=0.11, size=(per_batch, 1, hidden_size))
            class_indices = np.random.random_integers(0, 2, per_batch)
            class_indices = load_data.convert_to_one_hot(class_indices, 3) 
            words, loss = generative_predict_beam(gen_test, premises, noise_input,
                             class_indices, True, hypo_len)
            yield premises, words, loss, noise_input, class_indices

def validate(dev, gen_test, beam_size, hypo_len, samples, cmodel = None):
    vgen = val_generator(dev, gen_test, beam_size, hypo_len)
    p = Progbar(samples)
    while p.seen_so_far < samples:
        batch = next(vgen)
        preplexity = np.mean(np.power(2, batch[2]))
        loss = np.mean(batch[2])
        ceval = cmodel.evaluate([batch[0], batch[1]], batch[4], verbose = 0)
        
        p.add(len(batch[0]), [('loss',loss),('perplexity', preplexity),
                              ('class_loss', ceval[0]), ('class_acc', ceval[1])])


class ValidateGen(Callback):
    
    def __init__(self, dev, gen_test, beam_size, hypo_len, samples, cmodel):
        self.dev  = dev        
        self.gen_test=gen_test
        self.beam_size = beam_size
        self.hypo_len = hypo_len
        self.samples = samples
        self.cmodel= cmodel
    
    def on_epoch_end(self, epoch, logs={}):
        gm.update_gen_weights(self.gen_test[0], self.model)        
        validate(self.dev, self.gen_test, self.beam_size, self.hypo_len, self.samples,
                 self.cmodel)
    

                            

    
    
    
    

                             
    
    
