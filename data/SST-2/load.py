import json
import numpy as np
np.random.seed(1234)

class Corpus(object):
    def __init__(self, path):
        with open(path['dict']) as d:
            self.dictionary = json.load(d)

        self.train, self.train_mask, self.train_label = self.tokenize(path['train'])
        self.val, self.val_mask, self.val_label = self.tokenize(path['dev'])
        self.test, self.test_mask, self.test_label = self.tokenize(path['test'])

    def tokenize(self, path):
        sents=[]
        idxs=[]
        labels=[]
        with open(path) as f:
            for line in f:
                sent= line.split("\t")[0]
                idx= json.loads(line.split("\t")[2])
                label= int(line.split("\t")[1])
                sents.append(sent)
                idxs.append(idx)
                labels.append(label)

        lens = [len(s) for s in idxs]
        w = max(lens)
        h = len(idxs)
        mask = np.zeros((h, w), dtype='int32')
        data = np.zeros((h, w), dtype='int32')
        for i, s in enumerate(idxs):
            mask[i, :len(s)] = 1
            data[i, :len(s)] = s

        return data, mask, np.array(labels, dtype='int32')

data_file = {'dict':'dictionary.json', 'train':'train.json', 'dev':'dev.json', 'test':'test.json'}
corpus = Corpus(data_file)


idx_list = np.arange(corpus.train.shape[0], dtype='int')
np.random.shuffle(idx_list)
corpus.train = corpus.train[idx_list]
corpus.train_mask = corpus.train_mask[idx_list]
corpus.train_label = corpus.train_label[idx_list]

train_val_test_file = 'sst2.train_val_test_data.npz'

np.savez(train_val_test_file, alphabet=len(corpus.dictionary),
         train_x=corpus.train, train_m=corpus.train_mask, train_y=corpus.train_label,
         val_x=corpus.val, val_m=corpus.val_mask, val_y=corpus.val_label,
         test_x=corpus.test, test_m=corpus.test_mask, test_y=corpus.test_label)