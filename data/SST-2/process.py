import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np

# extract data from a csv
# notice the cool options to skip lines at the beginning
# and to only take data from certain columns
train_x=[]

train_y=[]

test_x=[]
test_y=[]

dev_x=[]
dev_y=[]
with open("train.tsv") as f:
   next(f)
   for line in f:
      line=line.strip()
      sent=line.split("\t")[0]
      label=line.split("\t")[1]
      train_x.append(sent)
      train_y.append(label)
with open("test.tsv") as f:
   next(f)
   for line in f:
      line=line.strip()
      idx=line.split("\t")[0]
      sent=line.split("\t")[1]
      test_x.append(sent)
      test_y.append(idx)
with open("dev.tsv") as f:
   next(f)
   for line in f:
      line=line.strip()
      sent=line.split("\t")[0]
      label=line.split("\t")[1]
      dev_x.append(sent)
      dev_y.append(label)
# create our training data from the tweets
# index all the sentiment labels

# only work with the 3000 most popular words found in our dataset
max_words = 3000

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words,oov_token="UNK")
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


allWordIndices = [t for t in  tokenizer.texts_to_sequences_generator(train_x)]
# for each tweet, change each token to its ID in the Tokenizer's word_index

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.

with open('train.json', 'w') as tjf: 
    for i,alw in enumerate(allWordIndices):
        tjf.write(train_x[i]+"\t"+train_y[i]+"\t"+json.dumps(alw)+"\n")



allWordIndices = []
allWordIndices = [t for t in  tokenizer.texts_to_sequences_generator(test_x)]
with open('test.json', 'w') as tjf: 
    for i,alw in enumerate(allWordIndices):
        tjf.write(test_x[i]+"\t"+test_y[i]+"\t"+json.dumps(alw)+"\n")

# feed our tweets to the Tokenizer
allWordIndices = []
allWordIndices = [t for t in  tokenizer.texts_to_sequences_generator(dev_x)]
# for each tweet, change each token to its ID in the Tokenizer's word_index
with open('dev.json', 'w') as tjf: 
    for i,alw in enumerate(allWordIndices):
        tjf.write(dev_x[i]+"\t"+dev_y[i]+"\t"+json.dumps(alw)+"\n")
