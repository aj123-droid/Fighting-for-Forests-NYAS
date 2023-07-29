#@title Import our libraries (this may take a minute or two)
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import spacy
import os # Good for navigating your computer's files 
import sys
pd.options.mode.chained_assignment = None #suppress warnings

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
# nltk.download('wordnet')
# nltk.download('punkt')

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import en_core_web_md
text_to_nlp = en_core_web_md.load()





tree_dataset = pd.read_csv('IMDB Dataset.csv')


columns = ['review', 'sentiment']
tree = tree_dataset[columns]

X_text = tree['review']
y = tree['sentiment']


def tokenize(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): 
            clean_tokens.append(token.lemma_)
    return clean_tokens
print('Hrllo')
bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X_text)
bow_transformer.vocabulary_
X = bow_transformer.transform(X_text)
pd.DataFrame(X.toarray())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
logistic_model = LogisticRegression()

logistic_model.fit(X_train,y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
print('Bienvenue')

# y_pred1 = logistic_model.predict(input('Enter text:'))