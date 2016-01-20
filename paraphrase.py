# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 13:24:47 2016

@author: Janez
"""


import load_data
import pickle
GLOVE = 'data/snli_vectors_300.txt'
PPDB = 'data/ppdb-2.0-s-all'


 


def test():
    result = set()    
    with open(PPDB) as ppdb:
        count = 0            
        for line in ppdb:
            example = line[:-1].split(' ||| ')
            p = example[1]
            h = example[2]
            rel = example[-1]
            result.add(rel)
            
            if rel == 'Equivalence' and p in glove:
                print p, h
               
            count += 1
            if count % 100000 == 0:
                break
    return result
    
def load_ppdb_data(glove):
    result = {}    
    with open(PPDB) as ppdb:
        count = 0            
        for line in ppdb:
            example = line[:-1].split(' ||| ')
            p = example[1]
            h = example[2]
            rel = example[-1]
            if rel == 'Equivalence':
                if not (p.startswith(h) or h.startswith(p)):       
                    if p in glove:
                        add_pair(result, p, h)
                    if h in glove:
                        add_pair(result, h, p)
                        
            count += 1
            if count % 100000 == 0:
                print count
    return result

def dump_parap(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def load_parap(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def add_pair(dct, p, h):
    if p not in dct:
      dct[p] = set()
    dct[p].add(h)
    
if __name__ == "__main__":
    glove = load_data.import_glove(GLOVE) 
    rep  =  load_ppdb_data(glove)  

