from keras.models import Sequential, Graph
from keras.layers.recurrent import LSTM
from keras.layers.core import Lambda, Dense
from keras import backend as K

from common import make_fixed_embeddings


def make_discriminator(glove, embed_size = 50, compile=False, hypo_len = 12):
    discriminator = Sequential()
    discriminator.add(make_fixed_embeddings(glove, hypo_len))
    discriminator.add(LSTM(embed_size))
    discriminator.add(Dense(1, activation='sigmoid'))
    if compile:
        discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

def make_full_adverse_model(discriminator, glove, embed_size = 100, batch_size = 64, hypo_len = 12):
    
    graph = Graph()
    graph.add_input(name='train_hypo', input_shape=(hypo_len,), dtype ='int')
    graph.add_input(name='gen_hypo', input_shape=(hypo_len,), dtype ='int')
    graph.add_shared_node(discriminator, name='shared', inputs=['train_hypo', 'gen_hypo'], merge_mode='join')
    
    def margin_opt(inputs):
        print(inputs)
        assert len(inputs) == 2, ('Margin Output needs '
                              '2 inputs, %d given' % len(inputs))
        u, v = inputs.values()
        return K.log(u) + K.log(1-v)
    
    graph.add_node(Lambda(margin_opt, output_shape = (1,)), name = 'output2', input='shared', create_output = True)
    graph.compile(loss={'output2':'mse'}, optimizer='adam')
    return graph
    
    
    
