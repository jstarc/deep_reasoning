

import generative_models
import generative_alg


batch_size = 64
hidden_size = 150
embed_size = 50
prem_len = 25

exp_suffix = 'ex1'

gen_train = generative_models.create_o2_train_model(len(train), hidden_size, embed_size, glove, batch_size, prem_len)
generative_alg.train_generative_graph(train, wi, gen_train, 'models/generative_' + exp_suffix, 
                                      nb_epochs = 100, batch_size = batch_size)
                                      
