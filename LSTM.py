import sys
import matplotlib.pyplot as plt
from sys import getsizeof
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import casual_tokenize
from fasttext import load_model
from inspect import getsource
import pickle
import copy
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, \
    Activation, SimpleRNN, Flatten, LSTM
from keras.metrics import categorical_accuracy, accuracy

# w2v = load_model('/home/ec2-user/nlp/cc.es.300.bin')
data = pd.read_pickle('./nlp/whatsapp_with_stop_words.pkl')

# 300-D cada vector
# TODO, es mejor hacer uno por uno la tokenizacion y vectorización
# TODO añadir el suffling del dataframe con sample frac=1

sentences = data.Sentence.apply(casual_tokenize).to_list()
labels = pd.get_dummies(data.Author).apply(list, axis=1)


indexes = list()
count = 0
sentences_w2v = list()
no_found = list()

for sentence in sentences:
    vectorized_sentence = list()
    for token in sentence:
        try:
            vectorized_sentence.append(w2v[token])
        except Exception as e:
            print(e)

    if(len(vectorized_sentence) > 0):
        sentences_w2v.append(vectorized_sentence)
        indexes.append(count)
    else:
        no_found.append(sentence)

    count += 1

# with open('./nlp/saved_states/sentences_vector.pkl', 'wb') as f:
#     pickle.dump(sentences_w2v, f)
#
# with open('./nlp/saved_states/indexes_df.pkl', 'wb') as f:
#     pickle.dump(indexes, f)

with open('./nlp/saved_states/sentences_vector.pkl', 'rb') as f:
    sentences_w2v = pickle.load(f)

with open('./nlp/saved_states/indexes_df.pkl', 'rb') as f:
    indexes = pickle.load(f)

temp_w2v = sentences_w2v.copy()


def pad_trunc(t, maxlen):
    new_data = []
    # Create a vector of 0s the length of our word vectors
    zero_vector = np.zeros(300)

    for sample in t:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample.copy()

            # Append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample.copy()
        new_data.append(temp)
    return new_data


# X = pad_trunc(temp_w2v, maxlen)
# X = np.array(X)
# y = np.array(labels.iloc[indexes].to_list())
#
# np.save('./nlp/saved_states/X.pkl', X)
# np.save('./nlp/saved_states/y.pkl', y)

X = np.load('./nlp/saved_states/X.pkl.npy')
y = np.load('./nlp/saved_states/y.pkl.npy')[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y)

maxlen = 15
batch_size = 32
embedding_dims = 300
num_neurons = 100
epochs = 100

model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True,
               input_shape=(maxlen, embedding_dims)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=100,
                    validation_data=(X_test, y_test))

model.summary()
plt.plot(history.history['accuracy'])
ax = plt.gca()
ax.plot(history.history['val_accuracy'])


sys.modules[__name__].__dict__.clear()


np.isna(np.nan)
