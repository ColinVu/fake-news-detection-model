# -*- coding: utf-8 -*-
"""

Fake News Detection Model - Created in 2021, revised in 2024
Colin Vu

"""

#@title Import Data { display-mode: "form" }
import math
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe

import pickle

!wget -O data.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Fake%20News%20Detection/inspirit_fake_news_resources%20(1).zip'
!unzip data.zip

basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))

"""Finding keywords from descriptions to understand article sentiment"""

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def scrape_description(url):
  if not url.startswith('http'):
    url = 'http://' + url
  response = requests.get(url, timeout=10)
  html = response.text
  description = get_description_from_html(html)
  return description

print('Description of Google.com:')
print(scrape_description('google.com'))

"""

Bag of Words: Turning each description into a set of words, then collecting the frequency of certain words to determine ones that are important and/or charged with positive/netative sentiment

"""

def get_descriptions_from_data(data):
  # Mapping from url to description for the websites in train_data.
  descriptions = []
  for site in tqdm(data):
    url, html, label = site
    descriptions.append(get_description_from_html(html))
  return descriptions

train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]

"""

Getting values from the descriptions and turning them into vectors

"""

val_descriptions = get_descriptions_from_data(val_data)

vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

print('\nPreparing train data')
bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_train_y = [label for url, html, label in train_data]

print('\nPreparing val data')
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_val_y = [label for url, html, label in val_data]

"""

Applying Logistic Regression to the bag of words (bow) model

"""

model = LogisticRegression()

bow_train_X = np.asarray(bow_train_X)
bow_train_y = np.asarray(bow_train_y)
bow_val_X = np.asarray(bow_val_X)

model.fit(bow_train_X, bow_train_y)
y_train_pred = model.predict(bow_train_X)
train_accuracy = accuracy_score(bow_train_y, y_train_pred)
y_val_pred = model.predict(bow_val_X)
val_accuracy = accuracy_score(bow_val_y, y_val_pred)
val_conf = confusion_matrix(bow_val_y, y_val_pred)
prf = precision_recall_fscore_support(bow_val_y, y_val_pred)

print("Train accuracy: " + str(train_accuracy))
print("Val accuracy: " + str(val_accuracy))
print("Val confusion matrix: " + str(val_conf))

print('Precision:', prf[0][1])
print('Recall:', prf[1][1])
print('F-Score:', prf[2][1])


"""

Application of a GloVe model to associate words with particular sentiment

"""

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

"""

Ex. Finding the vector for the word "good"

"""

good_vector = get_word_vector("good")
print('Shape of good vector:', good_vector.shape)
print(good_vector)

"""

Finding similarity in sentiment

"""

#@title Word Similarity { run: "auto", display-mode: "both" }

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

word1 = "more" #@param {type:"string"}
word2 = "some" #@param {type:"string"}

print('Word 1:', word1)
print('Word 2:', word2)

def cosine_similarity_of_words(word1, word2):
  vec1 = get_word_vector(word1)
  vec2 = get_word_vector(word2)

  if vec1 is None:
    print(word1, 'is not a valid word. Try another.')
  if vec2 is None:
    print(word2, 'is not a valid word. Try another.')
  if vec1 is None or vec2 is None:
    return None

  return cosine_similarity(vec1, vec2)


print('\nCosine similarity:', cosine_similarity_of_words(word1, word2))

"""

Descriptions -> array with GloVe vectors for each description

"""

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words

    return X

glove_train_X = glove_transform_data_descriptions(train_descriptions)
glove_train_y = [label for (url, html, label) in train_data]

glove_val_X = glove_transform_data_descriptions(val_descriptions)
glove_val_y = [label for (url, html, label) in val_data]

"""

Applying Logistic Regression

"""

model = LogisticRegression()
model.fit(glove_train_X, glove_train_y)
y_train_pred = model.predict(glove_train_X)
train_accuracy = accuracy_score(glove_train_y, y_train_pred)
y_val_pred = model.predict(glove_val_X)
val_accuracy= accuracy_score(glove_val_y, y_val_pred)
print("Train accuracy: " + str(train_accuracy))
print("Val accuracy: " + str(val_accuracy))

"""

Training and evaluating the models using the training data

"""

def train_model(train_X, train_y, val_X, val_y):
  model = LogisticRegression(solver='liblinear')
  model.fit(train_X, train_y)

  return model


def train_and_evaluate_model(train_X, train_y, val_X, val_y):
  model = train_model(train_X, train_y, val_X, val_y)

  y_train_pred = model.predict(train_X)
  y_val_pred = model.predict(val_X)
  train_accuracy = accuracy_score(train_y, y_train_pred)
  val_accuracy = accuracy_score(val_y, y_val_pred)
  val_conf = confusion_matrix(val_y, y_val_pred)
  prf = precision_recall_fscore_support(val_y, y_val_pred)
  print("Train accuracy: " + str(train_accuracy))
  print("Val accuracy: " + str(val_accuracy))
  print("Val confusion matrix: " + str(val_conf))
  print('Precision:', prf[0][1])
  print('Recall:', prf[1][1])
  print('F-Score:', prf[2][1])

  return model

"""

Combining the model evaluating keywords with a model evaluating domain name extensions

"""

def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        html = html.lower()
        y.append(label)
        features = featurizer(url, html)
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions

# Gets the count of keywords
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Mapping plaintext -> (url, html) pair.
def keyword_featurizer(url, html):
    features = {}

    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')

    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance'] # Can add more of these

    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)

    return features


keyword_train_X, keyword_train_y, features = prepare_data(train_data, keyword_featurizer)
keyword_val_X, keyword_val_y, features = prepare_data(val_data, keyword_featurizer)
train_and_evaluate_model(keyword_train_X, keyword_train_y, keyword_val_X, keyword_val_y)

vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(data_descriptions, vectorizer):
  X = vectorizer.transform(data_descriptions).todense()
  return X

bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)

bow_train_X = np.asarray(bow_train_X)
bow_train_y = np.asarray(bow_train_y)
bow_val_X = np.asarray(bow_val_X)
bow_val_y = np.asarray(bow_val_y)

model = train_and_evaluate_model(bow_train_X, bow_train_y, bow_val_X, bow_val_y)


VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words

    return X


train_X_des = glove_transform_data_descriptions(train_descriptions)
val_X_des = glove_transform_data_descriptions(val_descriptions)

model = train_model(train_X_des, glove_train_y, val_X_des, glove_val_y)
y_train_pred = model.predict(train_X_des)
y_val_pred = model.predict(val_X_des)
train_accuracy = accuracy_score(glove_train_y, y_train_pred)
val_accuracy = accuracy_score(glove_val_y, y_val_pred)
val_conf = confusion_matrix(glove_val_y, y_val_pred)
prf = precision_recall_fscore_support(glove_val_y, y_val_pred)
print("Train accuracy: " + str(train_accuracy))
print("Val accuracy: " + str(val_accuracy))
print("Val confusion matrix: " + str(val_conf))
print('Precision:', prf[0][1])
print('Recall:', prf[1][1])
print('F-Score:', prf[2][1])


"""

Combining all models

"""

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

combined_train_X = combine_features([
                                     bow_train_X, train_X_des, keyword_train_X, glove_train_X
])
combined_val_X = combine_features([
                                     bow_val_X, val_X_des, keyword_val_X, glove_val_X
])


model = train_and_evaluate_model(combined_train_X, y_train_pred, combined_val_X, y_val_pred)

"""

Final code allowing you to run the model on any website

"""

def get_data_pair(url):
  if not url.startswith('http'):
      url = 'http://' + url
  url_pretty = url
  if url_pretty.startswith('http://'):
      url_pretty = url_pretty[7:]
  if url_pretty.startswith('https://'):
      url_pretty = url_pretty[8:]

  response = requests.get(url, timeout=10)
  htmltext = response.text

  return url_pretty, htmltext

curr_url = "www.yahoo.com" #@param {type:"string"}

url, html = get_data_pair(curr_url)


def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X
def featurize_data_pair(url, html):
  # Keyword approach
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Description approach
  description = get_description_from_html(html)
  bow_X = vectorize_data_descriptions([description], vectorizer)
  # GloVe approach
  glove_X = glove_transform_data_descriptions([description])
  print(keyword_X.shape)
  print(bow_X.shape)
  print(glove_X.shape)
  X = combine_features([keyword_X, bow_X, glove_X, glove_X])

  return X

curr_X = np.array(featurize_data_pair(url, html))

model = train_model(combined_train_X, y_train_pred, combined_val_X, y_val_pred)
print(curr_X.shape)
curr_y = model.predict(curr_X)[0]


if curr_y < .5:
  print(curr_url, 'appears to be real.')
else:
  print(curr_url, 'appears to be fake.')
