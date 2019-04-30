from imdb import load_data, prepare_data
n_words = 10000
maxlen = 200
train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

train_x, train_mask, train_y = prepare_data(train[0], train[1], maxlen)
valid_x, valid_mask, valid_y = prepare_data(valid[0], valid[1], maxlen)
test_x, test_mask, test_y = prepare_data(test[0], test[1], maxlen)

train_val_test_file = ''.join(('./data/imdb/imdb.train_val_test_data.npz'))

import numpy as np
np.savez(train_val_test_file, alphabet=n_words,
         train_x=train_x, train_m=train_mask, train_y=train_y,
         val_x=valid_x, val_m=valid_mask, val_y=valid_y,
         test_x=test_x, test_m=test_mask, test_y=test_y)