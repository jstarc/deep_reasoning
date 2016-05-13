import numpy as np
import load_data
from generative_alg import *

def print_hypos(premise, label, gen_test, beam_size, hypo_len, noise_size, wi):
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
