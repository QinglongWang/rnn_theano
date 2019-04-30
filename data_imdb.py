from imdb import load_data, prepare_data
n_words = 10000
maxlen = 100
train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

x, mask, y = prepare_data(train, y)