import numpy as np
import load_data
from generative_alg import *
from keras.utils.generic_utils import Progbar
from load_data import load_word_indices
from keras.preprocessing.sequence import pad_sequences

def test_points(premises, labels, noises, gtest, cmodel, hypo_len):
    p = Progbar(len(premises))
    hypos = []
    bs = 64 
    for i in range(len(labels) / bs):
        words, _  = generative_predict_beam(gtest, premises[i * bs: (i+1)*bs], 
                          noises[i * bs: (i+1)*bs,None,:], labels[i * bs: (i+1)*bs], True, hypo_len)
        hypos.append(words)
        p.add(len(words))
    hypos = np.vstack(hypos)
    cpreds = cmodel.evaluate([premises[:len(hypos)], hypos], labels[:len(hypos)])
    print cpreds


def print_hypos(premise, label, gen_test, beam_size, hypo_len, noise_size, wi):
    words = single_generate(premise, label, gen_test, beam_size, hypo_len, noise_size)
    batch_size = gen_test[0].input_layers[0].input_shape[0]

    per_batch  = batch_size / beam_size
    premises = [premise] * per_batch
    noise_input = np.random.normal(scale=0.11, size=(per_batch, 1, noise_size))
    class_indices = np.ones(per_batch) * label
    class_indices = load_data.convert_to_one_hot(class_indices, 3)
    words, loss = generative_predict_beam(gen_test, premises, noise_input,
                             class_indices, True, hypo_len)
    
    print wi.print_seq(premise)
    print 
    for h in words:
        print wi.print_seq(h)

def load_premise(string, wi, prem_len = 25):
    tokens = string.split()
    tokens = load_word_indices(tokens, wi.index)
    return pad_sequences([tokens], maxlen = prem_len, padding = 'pre')[0]
    
