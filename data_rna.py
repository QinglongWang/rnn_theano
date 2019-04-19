import argparse
import os
import numpy as np

# python data.py --rna igf2bp123

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('--rna', type=str, default='taf2n', help = 'RNA data')
parser.add_argument('--seed', type=int, default= 123, help = 'seed for shuffling data')

args = parser.parse_args()
np.random.seed(args.seed)

def get_shuffled_ids(data, bsize):
    sorted_ids = np.argsort([len(l)+np.random.uniform(-1.0,1.0) for l in data])
    blocked_sorted_ids = [ sorted_ids[i:i+bsize] for i in xrange(0,len(data),bsize) ]
    np.random.shuffle( blocked_sorted_ids )
    return blocked_sorted_ids

class RNA_Corpus(object):
    def __init__(self, path):
        data = {'tr_pos':[], 'tr_neg':[], 'te_pos':[], 'te_neg':[]}
        pat_name = {'tr_pos':path[0], 'tr_neg':path[1], 'te_pos':path[2], 'te_neg':path[3]}
        self.alphabet = None
        self.max_len = 0

        for key in pat_name.keys():
            file_name = pat_name[key]
            data_raw = [l for l in open(file_name)]
            l = 0
            while l < len(data_raw):
                if data_raw[l][0] == '>':
                    i = 1
                    seq = []
                    while data_raw[l+i][0] != '>':
                        #if data_raw[l+i][0] == 'N':
                        #    i += 1
                        #    continue
                        seq += list(data_raw[l + i])[:-1]
                        i += 1
                        if l + i >= len(data_raw):
                            break
                    if len(seq) != 0:
                        data[key].append(seq)
                        if self.max_len < len(seq):
                            self.max_len = len(seq)
                    l += i
                else:
                    raise ValueError('Something wrong')

        self.train, self.train_mask, self.train_label = self.tokenize(data['tr_pos'], data['tr_neg'])
        self.test, self.test_mask, self.test_label = self.tokenize(data['te_pos'], data['te_neg'])

    def tokenize(self, pos, neg):
        if self.alphabet is None:
            alphabet_pos = sorted(set([a for l in pos for a in l]))
            alphabet_neg = sorted(set([a for l in neg for a in l]))
            self.alphabet = alphabet_pos
            if len(alphabet_pos) < len(alphabet_neg):
                self.alphabet = alphabet_neg

            self.dictionary = {a: i for i, a in enumerate(self.alphabet)}
        else:
            alphabet_pos = sorted(set([a for l in pos for a in l]))
            alphabet_neg = sorted(set([a for l in neg for a in l]))
            alphabet = alphabet_pos
            if len(alphabet_pos) < len(alphabet_neg):
                alphabet = alphabet_neg
            assert self.alphabet == alphabet

        data_all = pos + neg
        n = len(data_all)
        label = np.zeros(n, dtype='int32')
        label[:len(pos)] = 1
        mask = np.zeros((n, self.max_len), dtype='int32')
        data = np.zeros((n, self.max_len), dtype='int32')
        for i, s in enumerate(data_all):
            mask[i, :len(s)] = 1
            data[i, :len(s)] = [self.dictionary[a] for a in s]

        return data, mask, label

data_file = ['./data/RNA/' + args.rna + '.train.positive.fasta',
             './data/RNA/' + args.rna + '.train.negative.fasta',
             './data/RNA/' + args.rna + '.test.positive.fasta',
             './data/RNA/' + args.rna + '.test.negative.fasta']
train_val_test_file = ''.join(('./data/RNA/' + args.rna, '.train_val_test_data.npz'))

corpus = RNA_Corpus(data_file)

idx_list = np.arange(corpus.train.shape[0], dtype='int')
np.random.shuffle(idx_list)
corpus.train = corpus.train[idx_list]
corpus.train_mask = corpus.train_mask[idx_list]
corpus.train_label = corpus.train_label[idx_list]

np.savez(train_val_test_file, alphabet=corpus.alphabet,
         train_x=corpus.train, train_m=corpus.train_mask, train_y=corpus.train_label,
         test_x=corpus.test, test_m=corpus.test_mask, test_y=corpus.test_label)


