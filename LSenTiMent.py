#!/usr/bin/env python

__author__ = "Daniel del Castillo"
__version__ = "0.0.0"

"""
This is a baseline LSTM in Keras for sentence-level sentiment classification.
Tested for "glove.6B.300d.txt" as the word embeddings file and for a dataset
file with the format:
    [label0] [word00 word01 ...]
    [label1] [word10 word11 ...]
    ...
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
""" the above line eliminates the warning:
			"The TensorFlow library wasn't compiled to use SSE instructions,
			but these are available on your machine and could speed up CPU
			computations"
"""

import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.layers import Masking, Dense, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def load_data(embed_path, dataset_path, max_len, num_words):
    vec_list = []
    wd_to_id = {}
    id_to_wd = {}
    word_dict = {}
    # load word embeddings
    print('Loading word vectors...')
    with open(embed_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_splitted = line.strip().split(' ')
            wd_to_id[line_splitted[0]] = i
            id_to_wd[i] = line_splitted[0]
            vec_list.append(np.array([float(j) for j in line_splitted[1:]]))
            word_dict[line_splitted[0]] = vec_list[i]
            if i == num_words - 1:
                break
    print('Loaded {} {}-dimensional word vectors.'.format(i+1, len(vec_list[0])))
    embed_mat = np.array(vec_list)
    sentences = []
    num_unk_words = 0
    num_known_words = 0
    dataset_dict = {}
    X, Y = [], []
    # load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        #turn sentences into word vector sequences
        for line in f:
            line_splitted = line.strip().split(' ')
            label = line_splitted[0]
            words = line_splitted[1:]
            seq = []
            for w in words:
                if w in word_dict:
                    num_known_words += 1
                    seq.append(word_dict[w])
                    if w not in dataset_dict:
                        dataset_dict[w] = word_dict[w]
                else:
                    num_unk_words += 1
            if len(seq) > 0 and len(seq) <= max_len:
                X.append(seq)
                Y.append(label)
                sentences.append(words)
    print(num_unk_words, 'unknown words in the dataset.')

    return X, Y, embed_mat, wd_to_id, id_to_wd, dataset_dict, sentences

# >>>START>>>
embed_file = ''
dataset_file = ''
max_len_seq = 50
n_words = 400000
size_embed = 300
size_hidden = 128
nepochs = 10
batch_size = 64

# load dataset
X, Y, embedding_matrix, word_to_idx, idx_to_word, dataset_lexicon, sentence_list \
    = load_data(embed_file, dataset_file, max_len_seq, n_words)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create input and target arrays
input_data_train = np.zeros((len(X_train), max_len_seq, size_embed), dtype='float32')
target_data_train = np.zeros((len(Y_train), 1), dtype='float32')
input_data_test = np.zeros((len(X_test), max_len_seq, size_embed), dtype='float32')
target_data_test = np.zeros((len(Y_test), 1), dtype='float32')
# fill the arrays
for i, (x, y) in enumerate(zip(X_train, Y_train)):
    for t_inp, wordvec in enumerate(x):
        input_data_train[i,t_inp] = wordvec
    for t_tar, wordvec in enumerate(y):
        target_data_train[i,t_tar] = wordvec
for i, (x, y) in enumerate(zip(X_test, Y_test)):
    for t_inp, wordvec in enumerate(x):
        input_data_test[i,t_inp] = wordvec
    for t_tar, wordvec in enumerate(y):
        target_data_test[i,t_tar] = wordvec

# build model
print('Build model...')
model = Sequential()
input_mask = Masking(mask_value=0., input_shape=(None, size_embed))
lstm1 = LSTM(size_hidden, return_sequences=False, input_shape=(None, size_embed), activation='tanh')
sigmoid1 = Dense(1, activation='sigmoid')
model.add(input_mask)
model.add(lstm1)
model.add(sigmoid1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1, verbose=0, mode='auto')
print('Compile model...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(input_data_train, target_data_train,
          batch_size=batch_size,
          epochs=nepochs,
          validation_split=0.2,
          callbacks=[early_stop],
          verbose=1)

print('Evaluate...')
loss, acc = model.evaluate(input_data_test, target_data_test,
                batch_size=batch_size,
                verbose=1)

print('Evaluation results: loss: {} - acc: {}'.format(loss,acc))
