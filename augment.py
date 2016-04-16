import os
import glob
import numpy as np
import load_data
from generative_alg import generative_predict_beam
from adverse_alg import make_train_batch
from keras.utils.generic_utils import Progbar

def adversarial_generator(train, gen_model, discriminator, word_index, beam_size):
    batch_size, prem_len, _ = gen_model[0].inputs['premise'].input_shape
    examples = batch_size / beam_size
    hidden_size = gen_model[0].nodes['hypo_merge'].output_shape[2]
    hypo_len = discriminator.input_shape[1] 
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
                    'sanity': ad_preds, 'gen_probs' : probs}


def ca_generator(train, gen_model, discriminator, class_model, word_index, beam_size):
    ad_gen = adversarial_generator(train, gen_model, discriminator, 
                                   word_index, beam_size)
    batch = {}
    while True:
        ad_batch = next(ad_gen)
        class_batch = {'premise_input': ad_batch['premise'], 'hypo_input': ad_batch['hypo']}
        ad_batch['class_pred'] = class_model.predict_on_batch(class_batch)['output']  
        yield ad_batch

def generate_dataset(train, gen_model, discriminator, word_index, target_size,
                       beam_size, top_k = 1):

    iters = target_size / (3 * top_k)
    ad_gen = adversarial_generator(train, gen_model, discriminator, word_index, beam_size)
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

def pre_generate(train, gen_model, discriminator, class_model, word_index, 
                 beam_size, target_size):

    p = Progbar(target_size)
    ca_gen = ca_generator(train, gen_model, discriminator, class_model, word_index, 
                          beam_size)
    result_dict = {}
    while p.seen_so_far < target_size:
        batch = next(ca_gen)
        for k, v in batch.iteritems():
            result_dict.setdefault(k,[]).append(v)
        p.add(len(batch['hypo']))
    for k, v in result_dict.iteritems():
        result_dict[k] = np.concatenate(v)
    return result_dict

def pre_generate_save(train, gen_model, discriminator, class_model, word_index,
                      beam_size, target_dir, file_size = 30000):
    if not os.path.exists(target_dir):
         os.makedirs(target_dir)
    counter = 0
    while True:
        print 'Epoch', counter
        batch = pre_generate(train, gen_model, discriminator, class_model,
                                word_index, beam_size, file_size)
        filename = target_dir + '/data' + str(counter) 
        print_ca_batch(batch, word_index, filename)
        counter += 1

def deserialize_pregenerated(target_dir):
    import csv
    file_list = glob.glob(target_dir + '/*')
    dataset = []
    metrics = []
    for f in file_list:
        with open(f) as input:
            reader = csv.reader(input)
            header = next(reader)
            for ex in reader:
                dataset.append((ex[0].split(), ex[1].split(), ex[2]))
                metrics.append([float(ex[3]), float(ex[4])])
    return np.array(dataset), np.array(metrics)
             
def make_dataset(target_dir, train_len, dev_len):
    ds, met = deserialize_pregenerated(target_dir)
    label_len = (train_len + dev_len) / 3
    dev_label_len = dev_len / 3
    train = []
    dev = []
    for label in load_data.LABEL_LIST:
        indices = []
        for i in range(len(ds)):
            if ds[i][2] == label:
                indices.append(i)
        indices = np.array(indices)
        good_indices = indices[np.argsort(met[indices, 1])[-label_len:]]
        np.random.shuffle(good_indices)
        dev += list(ds[good_indices[:dev_label_len]])
        train += list(ds[good_indices[dev_label_len:]])
    return train, dev
    
    
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

def validate_generative(train, gen_model, discriminator, class_model, word_index, 
                        beam_size, target_size):
    data = pre_generate(train, gen_model, discriminator, class_model, word_index, 
                 beam_size, target_size)
    data_len = len(data['sanity'])
    cpred_loss =  -np.mean(np.log(data['class_pred'][np.arange(data_len), data['label']]))
    cpred_acc = np.mean(np.argmax(data['class_pred'], axis = 1) == data['label'])
    san_mean = np.mean(data['sanity'])
    gen_mean = np.mean(data['gen_probs'])
    
    return np.array([cpred_loss, cpred_acc, san_mean, gen_mean])

def test_gen_models(train, gen_train, gen_test, gen_folder, discriminator, class_model, word_index,
                    beam_size, target_size):

    model_list = glob.glob(gen_folder + '/*')
    model_list.sort(key=os.path.getmtime)
    for m in model_list:
        gen_train.load_weights(m)
        means = validate_generative(train, gen_test, discriminator, class_model, word_index,
                        beam_size, target_size)

        print m
        print means

