import numpy as np
import load_data
from generative_alg import *
from keras.utils.generic_utils import Progbar

def test_points(premises, labels, noises, gtest, cmodel, hypo_len):
    p = Progbar(len(premises))
    hypos = []
    for i in range(len(labels) / 16):
        words, _  = generative_predict_beam(gtest, premises[i * 16: (i+1)*16], noises[i * 16: (i+1)*16,None,:], labels[i * 16: (i+1)*16], True, hypo_len)
        hypos.append(words)
        p.add(len(words))
    hypos = np.vstack(hypos)
    cpreds = cmodel.evaluate([premises, hypos], labels)
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