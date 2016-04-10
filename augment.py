import numpy as np
import load_data
from generative_alg import generative_predict, generative_predict_beam
from adverse_alg import make_train_batch
from keras.utils.generic_utils import Progbar

def adversarial_generator(train, gen_model, discriminator, word_index, beam_size, hypo_len):
    batch_size, prem_len, _ = gen_model[0].inputs['premise'].input_shape
    examples = batch_size / beam_size
    hidden_size = gen_model[0].nodes['hypo_merge'].output_shape[2] 
    while True:
         mb = load_data.get_minibatches_idx(len(train), examples, shuffle=True)

         for i, train_index in mb:
             if len(train_index) != examples:
                 continue

             orig_batch = [train[k] for k in train_index]
             noise_input = np.random.normal(scale=0.11, size=(examples, 1, hidden_size))
             class_indices = np.random.random_integers(0, 2, examples)
             
             hypo_batch, probs = generative_predict_beam(gen_model, word_index, orig_batch,
                                            noise_input, class_indices, True, hypo_len)
             ad_preds = discriminator.predict_on_batch(hypo_batch)[0].flatten()
             
             X_prem, _, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
             premise_batch = load_data.pad_sequences(X_prem, maxlen = prem_len, dim = -1,
                                                     padding = 'pre')             
            
             yield {'premise' : premise_batch, 'hypo' : hypo_batch, 'label': class_indices,
                    'sanity': ad_preds}


def ca_generator(train, gen_model, discriminator, class_model, word_index, beam_size, 
                 batch_size, hypo_len):
    ad_gen = adversarial_generator(train, gen_model, discriminator, 
                                   word_index, beam_size, hypo_len)
    batch = {}
    class_batch_size = class_model.nodes['attention'].output_shape[0] ## check this
    while True:
        ad_batch = next(ad_gen)
        for k, v in ad_batch.iteritems():
            batch.setdefault(k,[]).append(v)
        temp_batches = batch[batch.keys()[0]]
        elements = len(temp_batches) * len(temp_batches[0])
        if elements == class_batch_size:
            for k, v in batch.iteritems():
                batch[k] = np.concatenate(v)
            class_batch = {'premise_input': batch['premise'], 'hypo_input': batch['hypo']}
            batch['class_pred'] = class_model.predict_on_batch(class_batch)['output']  
            yield batch
            batch = {}

def generate_dataset(train, gen_model, discriminator, word_index, target_size,
                       beam_size, hypo_len, top_k = 1):

    iters = target_size / (3 * top_k)
    ad_gen = adversarial_generator(train, gen_model, discriminator, word_index, beam_size,
                                   hypo_len)
    dataset = []
    max_scores = []
    p = Progbar(iters * 3 * top_k)
    for i in range(iters):
        ca_batch = next(ad_gen)
        scores = ca_batch['sanity']
        for l in range(3):
            ind = np.where(ca_batch['label'] == l)[0]
            top_rel_indices = np.argpartition(-scores[ind], min(top_k, len(ind) -1))[:top_k]
            top_indices = ind[top_rel_indices]            
            for t in top_indices:
                premise = word_index.get_seq(ca_batch['premise'][t].astype('int'))
                hypo = word_index.get_seq(ca_batch['hypo'][t].astype('int'))
                clss = load_data.LABEL_LIST[ca_batch['label'][t]]
                example = (premise, hypo, clss)
                dataset.append(example)
                max_scores.append(scores[t])
                p.add(1)
    return dataset, max_scores
    
        
                 
def print_ca_batch(ca_batch, wi, csv_file = None):
    
    writer = None
    
    if csv_file is not None:
        import csv
        csvf =  open(csv_file, 'wb')
        writer = csv.writer(csvf)
        writer.writerow(['premise', 'hypo', 'label', 'sanity'])#, 'class_prob'])

    for i in range(len(ca_batch[ca_batch.keys()[0]])):
        premise = wi.print_seq(ca_batch['premise'][i].astype('int'))
        hypo = wi.print_seq(ca_batch['hypo'][i].astype('int'))
        sanity = ca_batch['sanity'][i]
        label = load_data.LABEL_LIST[ca_batch['label'][i]]
        #class_prob = ca_batch['class_pred'][i][ca_batch['label'][i]]
   
        if csv_file is None:
            print premise
            print hypo
            print label, "sanity", sanity, 'cprob'#, class_prob
            print
        else:
            writer.writerow([premise, hypo, label, sanity])#, class_prob])                
    
    if csv_file is not None:
        csvf.close()
