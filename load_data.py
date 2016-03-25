import numpy as np
import json
import sys
sys.path.append('../keras')

DELIMITER = "--"
LABEL_LIST = ['neutral','contradiction','entailment']

def import_glove(filename, filter_set = None):
    word_map = dict()    
    with open(filename, "r") as f:
        for line in f:
            head, vec = import_glove_line(line)
            if filter_set == None or head in filter_set:      
                word_map[head] = vec
    return word_map

def write_glove(filename, glove):
    with open(filename, "w") as f:
        for head in glove:
            f.write(head + " " + ' '.join(np.char.mod('%.5g',glove[head])) + "\n")
        

def import_glove_line(line):
    partition = line.partition(' ')
    return partition[0], np.fromstring(partition[2], sep = ' ') 
    

def import_snli_file(filename):
    data = []   
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
    
def prepare_snli_dataset(json_data, add_null_token = False, exclude_undecided = True):
    dataset = []
    for example in json_data:
        sent1 = tokenize_from_parse_tree(example['sentence1_binary_parse'])
        sent2 = tokenize_from_parse_tree(example['sentence2_binary_parse'])
        gold = example['gold_label']
	if not exclude_undecided or gold in LABEL_LIST:
            if add_null_token:
                sent1 = ['null'] + sent1
            dataset.append((sent1, sent2, gold))
    return dataset
    
def tokenize_from_parse_tree(parse_tree):
    result = parse_tree.lower().replace('(', ' ').replace(')', ' ').split()
    result = ['(' if el=='-lrb-' else el for el in result]
    result = [')' if el=='-rrb-' else el for el in result]
    return result

def all_tokens(dataset):
    tokens = set()
    tokens.add(DELIMITER)    
    for e in dataset:
        tokens |= set(e[0])
        tokens |= set(e[1])
    return tokens
    

def repackage_glove(input_filename, output_filename, snli_path):
    train, dev, test = load_all_snli_datasets(snli_path)
    
    tokens = all_tokens(train) | all_tokens(dev) | all_tokens(test)
    glove = import_glove(input_filename, tokens)
    print "Glove imported"
    write_glove(output_filename, glove)

def load_all_snli_datasets(snli_path, add_null_token = False):
    print "Loading training data"
    train = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_train.jsonl'), add_null_token)
    print "Loading dev data"
    dev = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_dev.jsonl'), add_null_token)
    print "Loading test data"
    test = prepare_snli_dataset(import_snli_file(snli_path + 'snli_1.0_test.jsonl'), add_null_token)
    print "Data loaded"
    return train, dev, test

#repackage_glove('E:\\Janez\\Data\\vectors.6B.50d.txt', 'E:\\Janez\\Data\\snli_vectors.txt', 'E:\\Janez\\Data\\snli_1.0\\')


def prepare_vec_dataset(dataset, glove):
    X = []   
    y = []
    for example in dataset:
        if example[2] == '-':
            continue
        concat = example[0] + ["--"] + example[1]
        X.append(load_word_vecs(concat, glove))
        y.append(LABEL_LIST.index(example[2]))
    one_hot_y = np.zeros((len(y), len(LABEL_LIST)))
    one_hot_y[np.arange(len(y)), y] = 1
    return np.array(X), one_hot_y
    
def prepare_split_vec_dataset(dataset, word_index= None, glove = None):
    P = []
    H = []
    y = []
    for example in dataset:
        if example[2] == '-':
            continue
        if glove is not None:
            P.append(load_word_vecs(example[0], glove))
            H.append(load_word_vecs(example[1], glove))
        else:
            P.append(load_word_indices(example[0], word_index))   
            H.append(load_word_indices(example[1], word_index))
        y.append(LABEL_LIST.index(example[2]))
    
    one_hot_y = np.zeros((len(y), len(LABEL_LIST)))
    one_hot_y[np.arange(len(y)), y] = 1
    return np.array(P), np.array(H), one_hot_y
    
class WordIndex(object):
    def __init__(self, word_vec, eos_symbol = 'EOS'):
        self.keys =  word_vec.keys()
        index = self.keys.index(eos_symbol)
        self.keys[index], self.keys[0] = self.keys[0], eos_symbol 
        self.keys = np.array(self.keys)
        self.index = {key:value for key,value in zip(self.keys, range(len(self.keys)))}
    
    def print_seq(self, sequence):
        words = self.keys[sequence]
        words = [w for w in words if w != 'EOS']
        return " ".join(words)

    def get_seq(self, sequence):
        words = self.keys[sequence]
        return [w for w in words if w != 'EOS']

def load_word_vec(token, glove):
    if token not in glove:
	glove[token] = np.random.uniform(-1, 1, len(glove.values()[0]))    
    return glove[token]


def convert_to_one_hot(indices, vocab_size):
    return np.equal.outer(indices,np.arange(vocab_size)).astype(np.float)

#change this functions name
def prepare_one_hot_sents(dataset, glove_index, one_hot = True):
    H = []
    for s in dataset:
        
        sent_vec = np.zeros((len(s), len(glove_index))) if one_hot else np.zeros((len(s), 1))
        for i in range(len(s)):
            if one_hot:
                sent_vec[i][glove_index[s[i]]] = 1
            else:
                sent_vec[i][0] = glove_index[s[i]]
        H.append(sent_vec)
    return np.array(H)
    
    
def load_word_vecs(token_list, glove):
    return np.array([load_word_vec(x, glove) for x in token_list])   

def load_word_indices(token_list, word_index):
    return np.array([word_index[x] for x in token_list])     
        
def pad_sequences(sequences, maxlen=None, dim=1, dtype='float32',
    padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    dims = (nb_samples, maxlen) if dim < 0 else (nb_samples, maxlen, dim)
    x = (np.ones(dims) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def get_minibatches_idx_bucketing(lengths, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration accoring to lengths.
    """

    n= len(lengths)
    noise = np.random.random(n) - 0.5
    lengths += noise
    idx_list = np.argsort(lengths)
    

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    if shuffle:
	np.random.shuffle(minibatches)
    return zip(range(len(minibatches)), minibatches)

def get_minibatches_idx_bucketing_both(data, ranges, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration accoring to lengths.
    """
    
    idx_list = create_buckets(data, ranges)
    n = len(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    if shuffle:
        np.random.shuffle(minibatches)
    return zip(range(len(minibatches)), minibatches)

def create_buckets(data, ranges):
    result = [[] for x in range((len(ranges[0]) + 1) * (len(ranges[1]) + 1))]
    for e in range(len(data)):
        plen = len(data[e][0])
        hlen = len(data[e][1])
        pi = 0
        while pi < len(ranges[0]) and plen > ranges[0][pi]:
            pi += 1
        hi = 0
        while hi < len(ranges[1]) and hlen > ranges[1][hi]:
            hi += 1
        result[pi * (len(ranges[1]) + 1) + hi].append(e)
    master = []
    for r in result:
        np.random.shuffle(r)
        master += r
    return master

def get_minibatches_same_premise(data, minibatch_size):
    idx_list = np.arange(len(data), dtype="int32")
    prem_groups = {}
    for i in range(len(data)):
        premise = " ".join(data[i][0])
        if premise not in prem_groups:
            prem_groups[premise] = []
        prem_groups[premise].append(i)
    
    group_list = prem_groups.values()
     
    np.random.shuffle(group_list)
    minibatches = []
    minibatch = []
 
    for group in group_list:
        if len(minibatch + group) < minibatch_size:
            minibatch += group
        else:
            minibatches.append(np.array(minibatch))
            minibatch = group
    if len(minibatch) > 0:
        minibatches.append(np.array(minibatch))

    return zip(range(len(minibatches)), minibatches)

        
                
def prepare_word2vec(model, snli_path):
    train, dev, test = load_all_snli_datasets(snli_path)
    tokens = all_tokens(train) | all_tokens(dev) | all_tokens(test)
    glove = {}    
    for token in tokens:
        if token in model:
            glove[token] = model[token]
    return glove
        
def transform_dataset(dataset, class_str = None, max_prem_len = sys.maxint, max_hypo_len = sys.maxint):
    uniq = set()
    result = []
    for ex in dataset:
        prem_str = " ".join(ex[0])
        if class_str == None:
            prem_str += ex[2]
        if  (class_str == None or ex[2] == class_str) and prem_str not in uniq and len(ex[0]) <= max_prem_len and len(ex[1]) <= max_hypo_len:
            result.append(ex)
            uniq.add(prem_str)
    return result




if __name__ == "__main__":
    train, dev, test = load_all_snli_datasets('data/snli_1.0/')
    glove = import_glove('data/snli_vectors.txt')
    
    train = transform_dataset(train, None, 22, 12)
    dev = transform_dataset(dev, None, 22, 12)
    
    for ex in train+dev:
        load_word_vecs(ex[0] + ex[1], glove)
    #load_word_vec('EOS', glove)
    #glove['EOS'] = np.zeros(50)
    glove['EOS'] = np.array([
       -0.582148954417317066045711726474,
       -0.244393248315539324266865151003,
        0.741219472825796810155907223816,
       -0.733815277185625447486927441787,
        0.421104501955726684414571536763,
       -0.093848840526853827270770125324,
        0.26157600309975381769334035198 ,
       -0.83817126474407599445726191334 ,
        0.493374906303213345282188129204,
        0.885874879587261965241395955672,
        0.353202276464136488698386528995,
        0.857687901851755141180433383852,
       -0.30025513383325108662802449544 ,
        0.988116478739419656918130385748,
        0.622989339050785462248427393206,
       -0.601093506285039547165638396109,
        0.411576063562242744353625312215,
        0.606914919711505795874018076574,
        0.688334835361322117108784368611,
        0.520915135130978201871698729519,
        0.075241156713982126902351410536,
        0.665248424214184730374199716607,
        0.869791811730013852965726073307,
       -0.406030187847893131447563064285,
       -0.30600123632744780088899005932 ,
        0.009379291172922687991331258672,
       -0.668766958512569287265137063514,
        0.10997596322510072575084905111 ,
        0.293268911503105078608655276184,
        0.694830159357072929537935124245,
       -0.681338967971398368206337181618,
        0.698407201455419235358590412943,
        0.968944355797665135554552762187,
        0.691334715800610455360697415017,
       -0.651586373448130418140067376953,
        0.83215168062239741608721033117 ,
        0.826698989256635652367322109058,
        0.616670343805001408554744557478,
       -0.203716669102333769458823553578,
       -0.247300384492390668000894038414,
       -0.686098349518860617379800714843,
       -0.415813890692955601124936038104,
        0.72246883937489014826383026957 ,
        0.470699690746025911636252203607,
       -0.314883648927954729046518878022,
        0.591072134954823136254731252848,
        0.008658892001761842038831673563,
        0.999078362021354804411998884461,
        0.164330919435855093979625962675, 
       -0.066673396367238124682330635551])
    
    wi = WordIndex(glove)
        


    



        
        
        


