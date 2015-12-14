# -*- coding: utf-8 -*-
"""
@author: Janez Starc
"""

import load_data
import numpy as np

def predict_example(premise, hypothesis, model, glove):
    concat = premise.split() + ["--"] + hypothesis.split()
    vec = load_data.load_word_vecs(concat, glove)
    return model.predict_on_batch(np.expand_dims(vec, axis=0))

def create_error_analysis(dev, y_pred):
    with open('error_analysis.tsv', 'w') as file:
	file.write("Example\tpred\tgold\tneutral\tcontradiction\tentailment\n")
	for ex, y in zip (dev, y_pred):
	    text = " ".join(ex[0]) + " --> " + " ".join(ex[1])
	    row = [text, load_data.LABEL_LIST[np.argmax(y)], ex[2]] + [str(prob) for prob in y]
	    line = '\t'.join(row)
	    file.write(line + "\n")

        
