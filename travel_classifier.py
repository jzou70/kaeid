import json
import numpy as np

import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

questions = []
pairs = []
labels = []
data = json.load(open('travel_asq.json'))
for line in data:
	questions.append(line['Question'].lower())
	pairs.append((line['Question'].lower(),line['Single'].lower()))
	labels.append(line['Single'].lower())
cats = list(set(labels))

#map label to integer for NN classification
classToInt = {'numeric':0,'mc':1,'y/n':2,'location':3,'instructional':4,'entity':5}

#data for neural nets
train_x = questions
train_y = np.asarray([classToInt[x] for x in labels])

max_words = 3000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(train_x)
dictionary = tokenizer.word_index

def convert_text_to_array(text):
	return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allIndices = []
for text in train_x:
	allIndices.append(convert_text_to_array(text))
allIndices = np.asarray(allIndices)

train_x = tokenizer.sequences_to_matrix(allIndices, mode='binary')
train_y = keras.utils.to_categorical(train_y, 6)

#NEURAL NET CLASSIFIER
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1, validation_split=0.1, shuffle=True)

#NAIVE BAYES CLASSIFIERS
pipeline = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())
                    ])
pipeline2b = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())
                    ])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)
             }
gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
gs_clf2 = GridSearchCV(pipeline2b, parameters, n_jobs=-1)

#SVM CLASSIFIER
svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
pipeline3 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))
                    ])

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf-svm__alpha': (1e-2, 1e-3)
              }
gs_svm = GridSearchCV(pipeline3, parameters_svm, n_jobs=-1)  

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#SVM WITH LEMMATIZATION
pipeline4 = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))
                    ])

#EVALUATE
predicted = cross_val_predict(pipeline4,
						   questions,
						   labels,
						   cv=10)
print(cats)
print(confusion_matrix(labels,predicted,cats))
print(classification_report(labels,predicted))
