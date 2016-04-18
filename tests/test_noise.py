from keras.layers.core import Dense, Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers.embeddings import Embedding
import load_data

def noise_model(gen_train):
        
    input = Input(shape=(1,), dtype='int32')
    ne   = gen_train.layers[5]
    shape = ne.trainable_weights[0].shape.eval()
    emb = Embedding(shape[0], shape[1], input_length = 1, name='noise_embeddings',
         trainable = False, weights=[ne.get_weights()[0]])(input)   
    flt = Flatten()(emb)
    dense1 = Dense(150, activation = 'tanh')(flt)
    dense = Dense(3, activation='softmax')(dense1)
    
    model = Model(input=[input], output=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

def generator(train, word_index, batch_size, split = 500000, trainable =True):
    size = split if trainable else len(train) - split
   
    while True:
        mb = load_data.get_minibatches_idx(size, batch_size, shuffle=trainable)
        for _, train_index in mb:
            if not train:
                train_index += split
            _, _, y_train = load_data.prepare_split_vec_dataset(
                               [train[k] for k in train_index], word_index.index)
            yield [train_index], y_train

def train(model, train, word_index):
    split = 500000
    tgen = generator(train, word_index, 64, split, True)
    dgen = generator(train, word_index, 64, split, False)
    model.fit_generator(tgen, 25600, 20,  validation_data = dgen, 
                         nb_val_samples = len(train) - split)
    
    


