from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.models import Model

from attention import LstmAttentionLayer
from common import make_fixed_embeddings

def attention_model(hidden_size, glove):
        
    prem_input = Input(shape=(None,), dtype='int32')
    hypo_input = Input(shape=(None,), dtype='int32')
    
    prem_embeddings = make_fixed_embeddings(glove, None)(prem_input)
    hypo_embeddings = make_fixed_embeddings(glove, None)(hypo_input)
    premise_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid')(prem_embeddings)
    hypo_layer = LSTM(output_dim=hidden_size, return_sequences=True, 
                            inner_activation='sigmoid')(hypo_embeddings)    
    attention = LstmAttentionLayer(output_dim = hidden_size) ([premise_layer, hypo_layer])
    final_dense = Dense(3, activation='softmax')(attention)
    
    model = Model(input=[prem_input, hypo_input], output=final_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model  


