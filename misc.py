# -*- coding: utf-8 -*-
"""
@author: Janez Starc
"""

import load_data
import numpy as np
import operator

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
     
     
def subset_missing_glove_word(test_set):
    glove_bare = load_data.import_glove('data/snli_vectors_300.txt')
    result = []
    i = 0
    for ex in test_set:
	if not all(word in glove_bare for word in (ex[0] + ex[1])):
	    result.append(i)
	i += 1
    return result

def bucket_hypo_len(test_set, sent_ind, limits):
    lengths = [len(ex[sent_ind]) for ex in test_set]
    i = 0
    result = [[] for _ in range(len(limits) + 1)]
    for l in lengths:
        rank = 0
	for lim in limits:
	    if l <= lim:
		break
            rank += 1
	
	result[rank].append(i) 
    	i += 1
    return result

def wrong_index(y_pred, y_gold):
    return np.where(np.argmax(y_pred, axis=1) != np.argmax(y_gold, axis=1))[0]

def tfidf_on_wrong(texts, wrong):
    tf = {}
    idf = {}
    tfidf = {}
    for ex in np.array(texts)[wrong]:
	for word in set(ex[0] + ex[1]):
	    if word in tf:
		tf[word] += 1
	    else:
		tf[word] = 1

    for ex in texts:
        for word in set(ex[0] + ex[1]):
	    if word in idf:
                idf[word] += 1
            else:
                idf[word] = 1

    for word in idf:
	if word in tf:
	    tfidf[word] = tf[word] * len(texts) / float(idf[word]) / float(len(wrong))
    
    sorted_tfidf = sorted(tfidf.items(), key=operator.itemgetter(1))
    sorted_tfidf.reverse()
    final_list = [] 
    for tup in sorted_tfidf:
	el = tup[0] + ' ' + '{0:.2f}'.format(tup[1]) + ' ' + str(tf[tup[0]]) + ' ' + str(idf[tup[0]])
        final_list.append(el)

    return final_list

        
