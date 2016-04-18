

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import theano
theano.config.profile = True

gen_test = generative_models.create_o2_test_model(gen_train, glove)
gen_train.load_weights('models/generative_ex1/weights.09-0.17.hdf5')

generative_alg.generative_predict_beam(gen_test, wi, train[:16], np.random.normal(scale=0.11, size=(16, 1, 150)), np.random.random_integers(0, 2, 16), True, 15)
from theano.compile.profiling import _atexit_print_fn
_atexit_print_fn()



