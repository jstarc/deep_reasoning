from keras.layers.core import Dense, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.embeddings import Embedding
import load_data

def noise_model(gen_train):
        
    input = Input(shape=(1,), dtype='int32')
    ne   = gen_train.get_layer('noise_embeddings')
    shape = ne.trainable_weights[0].shape.eval()
    emb = Embedding(shape[0], shape[1], input_length = 1, name='noise_embeddings',
         trainable = False, weights=[ne.get_weights()[0]])(input)   
    flt = Flatten()(emb)
    dense1 = Dense(shape[-1], activation = 'tanh')(flt)
    dense = Dense(3, activation='softmax')(dense1)
    
    model = Model(input=[input], output=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

def generator(train, batch_size, split, trainable):
    size = split if trainable else len(train[0]) - split
    while True:
        mb = load_data.get_minibatches_idx(size, batch_size, shuffle=trainable)
        for _, train_index in mb:
            if not train:
                train_index += split
          
            yield [train_index], train[2][train_index]

def train(model, train):
    split = int(len(train[0]) * 0.95)
    tgen = generator(train, 64, split, True)
    dgen = generator(train, 64, split, False)
    model.fit_generator(tgen, 25600, 20,  validation_data = dgen, 
                         nb_val_samples = len(train[0]) - split)
    
    


def noise_test(gen_train):

    ne   = gen_train.get_layer('noise_embeddings')
    shape = ne.trainable_weights[0].shape.eval()

    noise_input = Input(shape=(shape[-1],))
    class_input = Input(shape=(3,))
    dense1 = Dense(shape[-1], activation = 'tanh')(merge([noise_input, class_input], mode = 'concat'))
    dense = Dense(1, activation='sigmoid')(dense1)

    model = Model(input=[noise_input, class_input], output=dense)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
