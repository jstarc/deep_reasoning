import os
import glob
import numpy as np
import load_data
from generative_alg import generative_predict_beam, val_generator
from common import merge_result_batches
from keras.utils.generic_utils import Progbar


def new_generate_dataset(dataset, samples, gen_test, beam_size, hypo_len, noise_size, cmodel):

    vgen = val_generator(dataset, gen_test, beam_size, hypo_len, noise_size)
    p = Progbar(samples)
    batchez = []
    while p.seen_so_far < samples:
        batch = next(vgen)
        probs = cmodel.predict([batch[0], batch[1]], verbose = 0)
        batch += (probs,)

        p.add(len(batch[0]))
        batchez.append(batch)
    return merge_result_batches(batchez)
   
def new_generate_save(dataset, target_dir, samples, gen_test, beam_size, hypo_len, noise_size, cmodel, 
                     word_index, name, target_size, threshold):
    if not os.path.exists(target_dir):
         os.makedirs(target_dir)
    counter = 1
    thresh_count = np.zeros(3)
    label_size = target_size / 3
    while (thresh_count < label_size).any():
        print 'Iteration', counter, thresh_count / label_size
        batch = new_generate_dataset(dataset, samples, gen_test, beam_size, hypo_len, noise_size, cmodel)
        cp = np.max(batch[5] * batch[4], axis = 1)
        for i in range(3):
            label_sum = np.sum((cp > threshold) * batch[4][:,i] == 1)
            thresh_count[i] += label_sum 
        filename = target_dir + '/' + name + str(counter)
        print_ca_batch(batch, word_index, filename)
        counter += 1


def deserialize_pregenerated(target_dir, wi):
    import csv
    file_list = glob.glob(target_dir + '/*')
    dataset ,losses, cpreds, ctrues = [],[],[],[]
    for f in file_list:
        with open(f) as input:
            reader = csv.reader(input)
            header = next(reader)
            for ex in reader:
                dataset.append((ex[0].split(), ex[1].split(), ex[2]))
                losses.append(float(ex[3]))
                cpreds.append(float(ex[4]))
                ctrues.append(ex[5] == 'True')
    from load_data import prepare_split_vec_dataset as prep_dataset
    return prep_dataset(dataset, wi.index, True) + (np.array(losses), np.array(cpreds))
        
    
def print_ca_batch(ca_batch, wi, csv_file = None):
    
    writer = None
    
    if csv_file is not None:
        import csv
        csvf =  open(csv_file, 'wb')
        writer = csv.writer(csvf)
        writer.writerow(['premise', 'hypo', 'label', 'loss', 'class_prob'])

    for i in range(len(ca_batch[0])):
        premise = wi.print_seq(ca_batch[0][i].astype('int'))
        hypo = wi.print_seq(ca_batch[1][i].astype('int'))
        loss = ca_batch[2][i]
        label = load_data.LABEL_LIST[np.argmax(ca_batch[4][i])]
        class_prob = ca_batch[5][i][np.argmax(ca_batch[4][i])]
        ctrue = (np.argmax(ca_batch[4][i]) == np.argmax(ca_batch[5][i]))
        if csv_file is None:
            print premise
            print hypo
            print label, "loss", loss, 'cprob', class_prob, 'ctrue', ctrue
            print
        else:
            writer.writerow([premise, hypo, label, loss, class_prob])                
    
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

def filter_dataset(dataset, threshold, final_size):
    eligible_args = dataset[5] if type(threshold) == bool else dataset[4] > threshold
    label_size = final_size / 3
    final_args = []
    for l in range(3):
        label_args = dataset[2][:, l] == 1
        intersect = label_args * eligible_args
        if np.sum(intersect) < label_size:
            print "loosen the constraints, label", l, np.sum(intersect) , label_size
            return None
        el_label_args = np.where(intersect)[0][:label_size]
        final_args += list(el_label_args)
    final_args = np.sort(final_args)
    list_dataset = [col[final_args] for col in dataset]
    return tuple(list_dataset)
    

    
    
    
    
