import load_data
import generative_models as gm
import generative_alg as ga
import classify_models as cm
import classify_alg as ca
import sys
import augment

if __name__ == "__main__":
    train, dev, test, wi, glove, prem_len, hypo_len = load_data.main()
    
    method = sys.argv[1]
    c_hidden_size = 150
    g_hidden_size = int(sys.argv[3]) 
    beam_size = 1
    version = int(sys.argv[2])
    batch_size = 64
    gen_epochs = 20
    latent_size = int(sys.argv[4])
    augment_file_size = 2 ** 15
    aug_threshold = 0.9

    epoch_size = (len(train[0]) / batch_size) * batch_size
    dev_sample_size = (len(dev[0]) / batch_size) * batch_size

    dir_name = 'models/real' + str(version) + '-' + str(g_hidden_size) + '-' + str(latent_size)

    cmodel = cm.attention_model(c_hidden_size, glove)
    cmodel.load_weights('models/cmodel/model.weights')

    if method == 'train_gen':
        gtrain = gm.gen_train(len(train[0]), g_hidden_size, latent_size, glove, hypo_len, version)
        ga.train(train, dev, gtrain, dir_name, batch_size, glove, beam_size, 
               epoch_size, dev_sample_size, cmodel, gen_epochs)

    if method == 'augment':
        gtrain = gm.gen_train(len(train[0]), g_hidden_size, latent_size, glove, hypo_len, version)
        gtrain.load_weights(dir_name + '/weights.hdf5')
        gtest = gm.gen_test(gtrain, glove, batch_size)
        augment.new_generate_save(train, dir_name, augment_file_size, gtest, beam_size, hypo_len, 
                                  latent_size, cmodel, wi, 'train', len(train[0]), aug_threshold)
        augment.new_generate_save(dev, dir_name, augment_file_size, gtest, beam_size, hypo_len,
                                  latent_size, cmodel, wi, 'dev', len(dev[0]), aug_threshold)
     
    if method == 'train_class':
       for t in thresholds:
           aug_train, aug_dev = augment.load_dataset(dir_name, t, len(train[0]), len(dev[0]), wi)
           aug_cmodel = cm.attention_model(c_hidden_size, glove)
           ca.train(aug_train, aug_dev, aug_cmodel, dir_name + '/threshold' + str(t), batch_size)
