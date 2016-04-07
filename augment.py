import numpy as np
import load_data
from generative_alg import generative_predict, generative_predict_beam
from adverse_alg import make_train_batch

def adversarial_generator(train, gen_model, discriminator, noise_embed_len, word_index, beam_size = 4, batch_size = 64, 
                          prem_len = 22, hypo_len = 12):
    examples = batch_size / beam_size 
    while True:
         mb = load_data.get_minibatches_idx(len(train), examples, shuffle=True)

         for i, train_index in mb:
             if len(train_index) != examples:
                 continue

             orig_batch = [train[k] for k in train_index]
             noise_input = np.random.normal(scale=0.11, size=(examples, 1, 50))
             class_indices = np.random.random_integers(0, 2, examples)
             
             hypo_batch, probs = generative_predict_beam(gen_model, word_index, orig_batch,
                                            noise_input, class_indices, True,
                                            batch_size, prem_len, hypo_len)
             ad_preds = discriminator.predict_on_batch(hypo_batch)[0].flatten()
             
             X_prem, _, _ = load_data.prepare_split_vec_dataset(orig_batch, word_index.index)
             premise_batch = load_data.pad_sequences(X_prem, maxlen = prem_len, dim = -1, padding = 'pre')             
            
             yield {'premise' : premise_batch, 'hypo' : hypo_batch, 'label': class_indices, 'sanity': ad_preds}


def ca_generator(train, gen_model, discriminator, class_model, noise_embed_len, word_index, beam_size = 4,
                 batch_size = 64, class_batch_size = 128, prem_len = 22, hypo_len = 12):
    ad_gen = adversarial_generator(train, gen_model, discriminator, noise_embed_len, 
                                   word_index, beam_size, batch_size, prem_len, hypo_len)
    batch = {}
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

def generate_dataset(train, gen_model, discriminator, class_model, noise_embed_len, word_index, target_size ,beam_size = 4,
                     batch_size = 64, class_batch_size = 128, prem_len = 22, hypo_len = 12):

    ca_gen = ca_generator(train, gen_model, discriminator, class_model, noise_embed_len, word_index, beam_size,
                          batch_size, class_batch_size, prem_len, hypo_len)
    dataset = []
    max_scores = []
    for i in range(target_size / 3):
        ca_batch = next(ca_gen)
        scores = ca_batch['sanity'] * ca_batch['class_pred'][np.arange(class_batch_size),ca_batch['label']]
        for l in range(3):
            ind = np.where(ca_batch['label'] == l)[0]
            max_ind = ind[np.argmax(scores[ind])]
            max_scores.append((ca_batch['sanity'][max_ind], ca_batch['class_pred'][max_ind][l], scores[max_ind]))
            premise = word_index.get_seq(ca_batch['premise'][max_ind].astype('int'))
            hypo = word_index.get_seq(ca_batch['hypo'][max_ind].astype('int'))
            clss = load_data.LABEL_LIST[ca_batch['label'][max_ind]]
            example = (premise, hypo, clss)
            dataset.append(example)
        print i, '\r' ,  
    return dataset#, max_scores
    
        
                 
def print_ca_batch(ca_batch, wi, csv_file = None):
    
    writer = None
    
    if csv_file is not None:
        import csv
        csvf =  open(csv_file, 'wb')
        writer = csv.writer(csvf)
        writer.writerow(['premise', 'hypo', 'label', 'sanity', 'class_prob'])

    for i in range(len(ca_batch[ca_batch.keys()[0]])):
        premise = wi.print_seq(ca_batch['premise'][i].astype('int'))
        hypo = wi.print_seq(ca_batch['hypo'][i].astype('int'))
        sanity = ca_batch['sanity'][i]
        label = load_data.LABEL_LIST[ca_batch['label'][i]]
        class_prob = ca_batch['class_pred'][i][ca_batch['label'][i]]
   
        if csv_file is None:
            print premise
            print hypo
            print label, "sanity", sanity, 'cprob', class_prob
            print
        else:
            writer.writerow([premise, hypo, label, sanity, class_prob])                
    
    if csv_file is not None:
        csvf.close()
