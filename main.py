import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
np.random.seed(7)

with open('xtrain_obfuscated.txt', 'r') as x_train_file:
    # x_train = x_train_file.readlines()
    x_train = x_train_file.read().splitlines()

with open('ytrain.txt', 'r') as y_train_file:
    y_train = y_train_file.read().splitlines()
y_train = list(map(int, y_train))
y_train = to_categorical(y_train)

n = len(x_train)
# find maximum and minimum length
vocab = set(''.join(x_train))
vocab_size = len(vocab)
mapping = {token: index for index, token in enumerate(vocab)}
# min_l = min(map(lambda x: len(x), x_train))
# max_l = max(map(lambda x: len(x), x_train))

# x_array = np.zeros(shape=(n,max_l, vocab_size))

# for i, sentence in enumerate(x_train):
#     for j, token in enumerate(sentence):
#         one_hot = mapping[token]
#         x_array[i, j, one_hot] = 1


sequences = list()
for sentence in x_train:
    encoded_line = [mapping[char] for char in sentence]
    sequences.append(encoded_line)


padded_sequences = pad_sequences(sequences, padding='post', value=-1)
# account for -1 token for padding
encoded_sequence = [to_categorical(sentence, num_classes=vocab_size+1) for sentence in padded_sequences]
x_array = np.array(encoded_sequence)


model = Sequential()
model.add(LSTM(128, input_shape=(x_array.shape[1], x_array.shape[2]), dropout=0.5))
# model.add(LSTM(128, return_sequences=True, dropout=0.5))
# model.add(LSTM(128, dropout=0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(x_array, y_train, epochs=100, verbose=1)