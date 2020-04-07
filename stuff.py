import re
import pandas as pd
from unidecode import unidecode
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# TODO hide the names and load them from CSV
# Remove the real names


def remove_stop_word(sentence, stopwords):
    temp = sentence.split(' ')
    temp = [x for x in temp if x not in stopwords]
    temp = ' '.join(temp)
    return temp


max_sequence_size = 1000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
embedding_dim = 16

sentences = []
author = []
current_author = []
names = ['Jose Cáliz', 'Cristhiam Sánchez Jaramillo',
         'Mateo Varela Martínez', 'Nicholas Benedetti Arévalo']

stopwords = pd.read_csv('./data_grupo/stopwords.csv')['word'].to_list()

expression_1 = r'\[.*?\]\s{0,}(.*?)\:\s{0,}(.*)'
re_expression_1 = re.compile(expression_1)

expression_2 = r'^[^\[].*'
re_expression_2 = re.compile(expression_2)

quote_expression = r'\u200e\[[0-9]{1,}\/[0-9]{1,}\/[0-9]{1,},.*'
re_quote = re.compile(quote_expression)

with open('./data_grupo/_chat.txt', 'r', encoding="utf-8") as f:
    for row in f:
        m_1 = re_expression_1.match(row)
        m_2 = re_expression_2.match(row)
        try:
            if(m_1):
                current_author = m_1.group(1)
                author.append(m_1.group(1))

                temp_sentence = m_1.group(2).lower()
                sentece_fix = remove_stop_word(temp_sentence, stopwords)
                sentences.append(sentece_fix)
            elif(m_2):
                author.append(current_author)

                temp_sentence = m_2.group(0).lower()
                sentece_fix = remove_stop_word(temp_sentence, stopwords)
                sentences.append(sentece_fix)
        except Exception as e:
            print(Exception)
            print(row)

data = pd.DataFrame({'Sentence': sentences, 'Author': author})
data = data[data.Author.isin(names)]
mask_not_quotes = [False if re_quote.match(x)
                   else True for x in data['Sentence']]
mask_not_empty = [False if len(x) == 0
                  else True for x in data['Sentence']]

data['quote'] = mask_not_quotes
data['empty'] = mask_not_empty

data = data[(data['quote'] & data['empty'])]

total_sentences = data['Sentence'].to_list()
total_labels = data['Author'].str.replace(' ', '').to_list()

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    total_sentences, total_labels, test_size=0.2, random_state=1234)

sentence_tokenizer = Tokenizer(num_words=40000, oov_token="<OOV>")
label_tokenizer = Tokenizer()

train_tokenizer.fit_on_texts(train_sentences)
label_tokenizer.fit_on_texts(train_labels)

word_index_sentence = train_tokenizer.word_index
word_index_label = label_tokenizer.word_index

train_sentences_sequences = train_tokenizer.texts_to_sequences(train_sentences)
test_sentences_sequences = train_tokenizer.texts_to_sequences(test_sentences)
train_labels_sequences = label_tokenizer.texts_to_sequences(train_labels)
test_labels_sequences = label_tokenizer.texts_to_sequences(test_labels)

train_sentences_padded = pad_sequences(
    train_sentences_sequences,
    maxlen=max_sequence_size,
    truncating=trunc_type,
    padding=padding_type
)

test_sentences_padded = pad_sequences(
    test_sentences_sequences,
    maxlen=max_sequence_size,
    truncating=trunc_type,
    padding=padding_type
)

vocab_size = len(word_index_sentence.keys())
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              input_length=max_sequence_size),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

num_epochs = 10
training_padded = np.array(train_sentences_padded)
training_labels = np.array(train_labels_sequences)
testing_padded = np.array(test_sentences_padded)
testing_labels = np.array(test_labels_sequences)


def genearte_array(i):
    t = np.zeros(4)
    t[i-1] = 1
    return t


matrix_train = np.array([genearte_array(train_labels_sequences[i][0]) for i in
                         range(len(train_labels_sequences))])
matrix_test = np.array([genearte_array(test_labels_sequences[i][0]) for i in
                        range(len(test_labels_sequences))])

history = model.fit(training_padded, matrix_train, epochs=num_epochs,
                    validation_data=(testing_padded, matrix_test),
                    verbose=1)
