import generative_models
gen_model = generative_models.create_o_train_model(len(train), 150, 50, glove)
gen_model.load_weights('models/new_gen_model/weights.12-0.36.hdf5')
gen_test = generative_models.create_o_test_model(gen_model, len(train), 150, 50, glove)
import adverse_models
discriminator = adverse_models.make_discriminator(glove, 150, compile=False)
ad_model = adverse_models.make_full_adverse_model(discriminator, glove, 150)

