import numpy as np
import load_data
from generative_alg import generative_predict
from adverse_alg import make_train_batch

def adversarial_generator(train, gen_model, discriminator, noise_embed_len, word_index, batch_size = 64, 
                          prem_len = 22, hypo_len = 12):
    while True:
         mb = load_data.get_minibatches_idx(len(train), batch_size, shuffle=True)

         for i, train_index in mb:
             if len(train_index) != batch_size:
                 continue

             orig_batch = [train[k] for k in train_index]
             noise_input = np.random.random_integers(0, noise_embed_len -1, (len(orig_batch), 1))
             class_indices = np.random.random_integers(0, 2, len(orig_batch))

             probs = generative_predict(gen_model, word_index.index, orig_batch, noise_input, class_indices,
                                            batch_size, prem_len, hypo_len)
             hypo_batch = np.argmax(probs, axis = 2)
             ad_preds = discriminator.predict_on_batch(hypo_batch)[0].flatten()
             
             X_prem, _, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
             premise_batch = load_data.pad_sequences(X_prem, maxlen = prem_len, dim = -1, padding = 'pre')             
            
             yield {'premise' : premise_batch, 'hypo' : hypo_batch, 'label': class_indices, 'sanity': ad_preds}


def ca_generator(train, gen_model, discriminator, class_model, noise_embed_len, word_index, batch_size = 64,
                 class_batch_size = 128, prem_len = 22, hypo_len = 12):
    ad_gen = adversarial_generator(train, gen_model, discriminator, noise_embed_len, 
                                   word_index, batch_size, prem_len, hypo_len)
    batch = {}
    while True:
        ad_batch = next(ad_gen)
        for k, v in ad_batch.iteritems():
            batch.setdefault(k,[]).append(v)
             
        if len(batch[batch.keys()[0]]) == class_batch_size / batch_size:
            for k, v in batch.iteritems():
                batch[k] = np.concatenate(v)
            class_batch = {'premise_input': batch['premise'], 'hypo_input': batch['hypo']}
            batch['class_pred'] = class_model.predict_on_batch(class_batch)['output']  
            yield batch
            batch = {}
                 
def print_ca_batch(ca_batch, wi):
    for i in range(len(ca_batch[ca_batch.keys()[0]])):
        premise = wi.print_seq(ca_batch['premise'][i].astype('int'))
        hypo = wi.print_seq(ca_batch['hypo'][i].astype('int'))
        sanity = ca_batch['sanity'][i]
        label = load_data.LABEL_LIST[ca_batch['label'][i]]
        class_prob = ca_batch['class_pred'][i][ca_batch['label'][i]]
   
        print premise
        print hypo
        print label, "sanity", sanity, 'cprob', class_prob
        print
    
