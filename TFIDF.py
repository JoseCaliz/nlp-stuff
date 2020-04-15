import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.tokenize.casual import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import BernoulliNB

# TODO hide the names and load them from CSV
# Remove the real names


def remove_stop_word(sentence, stopwords):
    temp = sentence.split(' ')
    temp = [x for x in temp if x not in stopwords]
    temp = ' '.join(temp)
    return temp


names = ['Jose Cáliz', 'Cristhiam Sánchez Jaramillo',
         'Mateo Varela Martínez', 'Nicholas Benedetti Arévalo']

stopwords = pd.read_csv('./data_grupo/stopwords.csv')['word'].to_list()

expression_1 = r'\[.*?\]\s{0,}(.*?)\:\s{0,}(.*)'
re_expression_1 = re.compile(expression_1)

expression_2 = r'^[^\[].*'
re_expression_2 = re.compile(expression_2)

quote_expression = r'\u200e\[[0-9]{1,}\/[0-9]{1,}\/[0-9]{1,},.*'
re_quote = re.compile(quote_expression)

sentences = list()
author = list()

with open('./data_grupo/_chat.txt', 'r', encoding="utf-8") as f:
    count = 0
    for row in f:
        m_1 = re_expression_1.match(row)
        m_2 = re_expression_2.match(row)
        try:
            if(m_1):
                current_author = m_1.group(1)
                author.append(m_1.group(1))

                temp_sentence = m_1.group(2).lower()
                sentences.append(temp_sentence)
            elif(m_2):
                author.append(current_author)

                temp_sentence = m_2.group(0).lower()
                sentences.append(temp_sentence)
        except Exception as e:
            print(Exception)
            print(row)

data = pd.DataFrame({'Sentence': sentences, 'Author': author})
data = data[data.Author.isin(names)]
mask_not_quotes = [False if re_quote.match(x)
                   else True for x in data['Sentence']]
mask_not_empty = [False if len(x) == 0
                  else True for x in data['Sentence']]

# Remove messages that are quotes
data['quote'] = mask_not_quotes
data['empty'] = mask_not_empty

data = data[(data['quote'] & data['empty'])]

# Get the labels for each author
total_labels = data['Author'].str.replace(' ', '').to_list()
classnames, indices = np.unique(total_labels, return_inverse=True)
data['label'] = indices

data.head(10)
messages = data['Sentence'].to_list()
label = data['label'].to_list()

# Prueba con LDiA
counter = CountVectorizer(tokenizer=casual_tokenize)
bow_vector = pd.DataFrame(counter.fit_transform(raw_documents=messages)
                          .toarray())

# Extaer el vocabulario para ponérselo a las columnas de bow_vector
vocabulario = counter.vocabulary_.values()
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),
                                     counter.vocabulary_.keys())))
bow_vector.columns = terms
ldia = LDiA(n_components=100,  learning_method='batch')
topic_vectors = ldia.fit_transform(bow_vector)

# Split
results_train = []
results_test = []
for i in range(4):
    label = [1 if x == i else 0 for x in data['label']]
    X_train, X_test, y_train, y_test = train_test_split(topic_vectors, label,
                                                        test_size=0.3)

    # Do the LDA
    # lda = LDA(n_components=1)
    lda = BernoulliNB()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    results_train.append(round(float(lda.score(X_train, y_train)), 2))
    results_test.append(round(float(lda.score(X_test, y_test)), 2))

print(results_train)
print(results_test)

results = dict(zip(classnames, results))
results
