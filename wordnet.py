# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:10:52 2016

@author: Janez
"""

import re

def proc_data(filename):
    with open(filename, 'r') as f:
        count = 0    
        synsets = {}
        for line in f:
            triple = parse_triple(line)
            if triple is not None:
               subj,pred,obj = triple 
               if subj not in synsets:
                    synsets[subj] = Synset(subj)
               syn = synsets[subj]
               if pred == 'label':
                   syn.add_label(obj)
               else:
                   syn.add_rel(pred, obj)
               
            count += 1
            if count % 100000 == 0:
                print count
        return synsets
        
def strip_url(argument):
    if argument[0] == '<' and argument[-1] == '>':
        return argument[1:-1]
    else:
        print "Not url", argument
        return argument
        
def get_last_split(url, delimiter):
    splits = url.split(delimiter)
    if len(splits) <= 1:
        print "Unexpected input", url
        return url
    else:
        return splits[-1]

def parse_string(string):
    if '@' not in string:
        return None
    content, lang = string.split('@')
    if  content[0] == '"' and content[-1] == '"':
        return content[1:-1], lang
        
def get_type(argument):
    if argument[0] == '<':
        return 'url'
    elif argument[0] == '"':
        return 'string'
    else:
        print 'Unknown arg', argument
        return 'unknown'

def parse_triple(triple):
    splits = triple.split(' ')
    if len(splits) < 4:
        if len(triple.strip()) > 0:
            print 'split failed', len(triple), triple        
        return None
    
    if get_type(splits[0]) != 'url':
        print "subj not url", splits[0]
        return None
    subj = get_last_split(strip_url(splits[0]), '/')
    
    if get_type(splits[1]) != 'url':
        print "pred not url", splits[1]
        return None
    pred = get_last_split(strip_url(splits[1]), '#')
    
    raw_obj = " ".join(splits[2: -1])
    if get_type(raw_obj) == 'string':
        if parse_string(raw_obj) is None:
            return None        
        content, lang = parse_string(raw_obj)
        if lang == "eng":
            obj = content
        else:
            return None
    elif get_type(raw_obj) == 'url':
        url = strip_url(raw_obj)       
        if '#' in url:
            obj = get_last_split(url, '#')
        else:
            obj = get_last_split(url, '/')
    else:
        return None
            
    return subj, pred, obj
    
def create_synonym_label_map(syn_map):
    label_map = {}
    for syn in syn_map.values():
        if len(syn.labels) < 2:
            continue
        for label1 in syn.labels:
            for label2 in syn.labels:
                if label1 != label2:
                    if label1.lower() not in label_map:
                        label_map[label1.lower()] = set()
                    label_map[label1.lower()].add(label2.lower())
    return label_map
    
def create_hypernym_map(syn_map):
    label_map = {}
    for syn in syn_map.values():
        if syn.relation.get('hypernym') and len(syn.labels) > 0:
            for hypname in syn.relation['hypernym']:
                if hypname in syn_map and len(syn_map[hypname].labels) > 0:
                    for label1 in syn.labels:
                        for label2 in syn_map[hypname].labels:
                            if label1.lower() not in label_map:
                                label_map[label1.lower()] = set()
                            label_map[label1.lower()].add(label2.lower()) 
    return label_map

def create_hypernym_map2(lab_map, syn_map):
    result = {}
    for lab in lab_map:
        for syn in lab_map[lab]:
            if syn.relation.get('hypernym') and len(syn.labels) > 0:
                for hypname in syn.relation['hypernym']:
                    if hypname in syn_map and len(syn_map[hypname].labels) > 0:
                        for label2 in syn_map[hypname].labels:
                            if lab.lower() not in result:
                                result[lab.lower()] = set()
                            result[lab.lower()].add(label2.lower()) 
    return result
    
def create_label_to_concept_map(syns, pos_filter=''):
    mapping = {}
    for s in syns:
        syn = syns[s]
        if len(pos_filter) > 0 and 'part_of_speech' in syn.relation \
            and syn.relation['part_of_speech'][0] == pos_filter:
                for lab in syn.labels:
                    if lab not in mapping:
                        mapping[lab.lower()] = []
                    mapping[lab.lower()].append(syn)
    return mapping

def filter_lab_map(lab_map):
    new_map = {}    
    for lab in lab_map:
        if len(lab_map[lab]) > 1:
            new_map[lab] = filter_syn_mapping(lab, lab_map[lab])
        else:
            new_map[lab] = lab_map[lab]
    return new_map
    
def filter_syn_mapping(label, syns):
    bmap = [True] * len(syns)
    for i in range(len(syns)):
        s = syns[i]
        if 'type' not in s.relation or s.relation['type'][0] != 'Synset':
           if mark_bit_map(bmap, i):
               break
        if not is_informative_same_as(s, label):
            if mark_bit_map(bmap, i):
               break
    return [i for (i, v) in zip(syns, bmap) if v]
         
def is_informative_same_as(syn, label):
    if 'sameAs' in syn.relation:
        for s2 in syn.relation['sameAs']:
            pattern = re.compile('^synset-'+ label +'-[a-z]*-1$')
            if pattern.match(s2):
                return True
    return False
    
def lab_map_to_syn_map(lab_map):
    syn_map = {}
    for syn_list in lab_map.values():
        for syn in syn_list:
            syn_map[syn.name] = syn
    return syn_map
            
        
def mark_bit_map(bmap, index):
    if sum(bmap) > 1:
        bmap[index] = False
        return False
    return True


            
    
def match_string(list_string, mapping, substring_len):
    matches = []
    for i in range(min(substring_len, len(list_string))):
        for j in range(len(list_string) - i):
            substr = " ".join(list_string[j:(j+i+1)])
            if substr in mapping:
                matches.append(substr)
    return matches            


    

    
class Synset(object):
    def __init__(self, name):
        self.name = name
        self.labels = []
        self.relation = {}
        
    def add_label(self, label):
        self.labels.append(label)
        
    def add_rel(self, pred, obj):
        if pred not in self.relation:
            self.relation[pred] = []
        self.relation[pred].append(obj)
    
    def __str__(self):
        string = "Name: " + self.name + "\n"
        string += "Labels: " +  str(self.labels) + "\n"
        for rel in self.relation:
            string += rel + str(self.relation[rel]) + "\n"
        return string
        
        
    
        
    
    
    
    
    
            
        