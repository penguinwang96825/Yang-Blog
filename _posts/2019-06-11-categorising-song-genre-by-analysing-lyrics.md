---
layout: post
title: Categorising Song Genre by Analysing Lyrics
subtitle: Categorising Song Genre by Analysing Lyrics
cover-img: /assets/img/2019-06-11-categorising-song-genre-by-analysing-lyrics/hanny-naibaho.jpg
thumbnail-img: /assets/img/2019-06-11-categorising-song-genre-by-analysing-lyrics/14th.jpg
readtime: true
show-avatar: false
tags: [Python, NLP, Word2Vec, GloVe, FastText, BERT]
comments: true
---

The ability to classify music in an automated manner has become increasingly more important with the advent of musical streaming services allowing greater access to music. Spotify alone hit 100 million users in 2016, with other services provided by companies such as Apple, Soundcloud and YouTube. In addition, there are huge numbers of professional musicians, approximately 53,000 in the USA alone, as well as amateurs who are producing music which needs to be classified. With this quantity of music, it is unfeasible to classify genres without an automated method.

# Introduction

The aim of this project is to try to develop a classifier for song genres using only its lyrics. Firstly, a dataset of song lyrics and their associated genres needs to be produced. Therefore, I build a crawler to get the dataset, which I will not demonstate in this article. Secondly, a review of the potential classification models needs to be undertaken to determine which is most likely to be successful in this task. I will compare conventional machine learning models to state-of-the-art deep learning models. Thirdly, a final result should be produced with the optimised model. This will then be reviewed with comparison to both ML and DL models to determine what areas are working successfully and where there are remaining issues, which still need to be overcome.

# Exploratory Data Analysis

## Import Libraries

```python
import re
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")

import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
```

---

Load the data.

```python
data = pd.read_csv("./lyrics.csv", delimiter=",")
data.head()
```

![](/assets/img/2019-06-11-categorising-song-genre-by-analysing-lyrics/data.png)

## Data Processing

There are lots of techniques for NLP data processing, such as noise removal (remove stopwords), lexicon normalisation (stemming, lemmatisation), object standardisation (acronyms, hash tags, colloquial slangs), etc. However, these are out of the scope in this article, instead, I will only do tokenisation. Maybe I will post another article discussing about different kind of text processing procedure in the future.

```python
def tokenization(text):
    text = re.split('\W+', text)
    return text

data["lyrics_tokenised"] = data["lyrics"].apply(lambda x: tokenization(str(x).lower()))
```

Segregated data into training (40%), validation (20%), and testing (20%) dataset.

```python
lyrics = data['lyrics_tokenised'].values
genres = data['genre'].apply(str).values

X_train, X_test, y_train, y_test = train_test_split(
	lyrics, genres, test_size=0.4, random_state=914, stratify=genres)
X_valid, X_test, y_valid, y_test = train_test_split(
	X_test, y_test, test_size=0.5, random_state=914, stratify=y_test)
```

There are 217342, 72447, 72447 samples of training, validation, and testing dataset, respectively.


# Word Embedding

I utilised three word embedding vectorisers `MeanEmbeddingVectorizer()`, `TfidfEmbeddingVectorizer()`, and `SifEmbeddingVectorizer()` from one of my [post]({{site.baseurl}}{% link _posts/2021-01-25-weighted-word-embedding.md %}).

```python
# Word2Vec
vectoriser_w2v_mean = MeanEmbeddingVectorizer(word2vec=w2v_model)
feature_train_w2v_mean = vectoriser_w2v_mean.fit_transform(X_train, None)
vectoriser_w2v_tfidf = TfidfEmbeddingVectorizer(word2vec=w2v_model)
feature_train_w2v_tfidf = vectoriser_w2v_tfidf.fit_transform(X_train, None)
vectoriser_w2v_sif = SifEmbeddingVectorizer(word2vec=w2v_model)
feature_train_w2v_sif = vectoriser_w2v_sif.fit_transform(X_train, None)

# GloVe
vectoriser_glove_mean = MeanEmbeddingVectorizer(word2vec=glove_model)
feature_train_glove_mean = vectoriser_glove_mean.fit_transform(X_train, None)
vectoriser_glove_tfidf = TfidfEmbeddingVectorizer(word2vec=glove_model)
feature_train_glove_tfidf = vectoriser_glove_tfidf.fit_transform(X_train, None)
vectoriser_glove_sif = SifEmbeddingVectorizer(word2vec=glove_model)
feature_train_glove_sif = vectoriser_glove_sif.fit_transform(X_train, None)

# FastText
vectoriser_ft_mean = MeanEmbeddingVectorizer(word2vec=ft_model)
feature_train_ft_mean = vectoriser_ft_mean.fit_transform(X_train, None)
vectoriser_ft_tfidf = TfidfEmbeddingVectorizer(word2vec=ft_model)
feature_train_ft_tfidf = vectoriser_ft_tfidf.fit_transform(X_train, None)
vectoriser_ft_sif = SifEmbeddingVectorizer(word2vec=ft_model)
feature_train_ft_sif = vectoriser_ft_sif.fit_transform(X_train, None)
```

After a long period of time, finally got the embedding vectors! Let's put them into dictionaries `vectorisers_dcit` and `features_train_dict` for later use.

# Modelling

## Introduction

This is a supervised text classification problem, and our goal is to investigate which supervised machine learning methods are best suited to solve it. Given a new lyrics comes in, we want to assign it to one of the twelve categories. This is a multi-class text classification task. 

## Imbalanced Classes

Let's take a look at the distribution of label in training dataset.

![](/assets/img/2019-06-11-categorising-song-genre-by-analysing-lyrics/distribution.png)

We can see that the number of genres per song is imbalanced. Genres of the songs are more biased towards "Rock" music. When we encounter such problems, we are bound to have difficulties solving them with standard algorithms, Conventional algorithms are often biased towards the majority classes, not taking the data distribution into account. In the worst case, minority classes are considered as outliers or being ignored. For some cases, such as fraud detection or cancer prediction, we would need to carefully configure our model or artificially balance the dataset, for instance, using resampling technique (under-sampling, over-sampling), Tomek Links, SMOTE (Synthetic Minority Oversampling Technique), class weights in the models, or changing your evaluation metrics.

Various other methods might work depending on your use case and the problem you are trying to solve. 
1. Collect more data
2. Treat the problem as anomaly detection (e.g. isolation forests, autoencoders, ...)
3. Model-based approach (boosting models, ...)

However, in our case, I will not operate any of the techniques mentioned above, I will leave it as it is.

## Baseline

We are now ready to experiment with different machine learning models, evaluate their accuracy and find the source of any potential issues.

We will benchmark the following three models:
* Random Forest
* Linear Support Vector Machine
* Logistic Regression

| Model         | Fold          | Acc  |
| ------------- |:-------------:| -----:|
| RandomForestClassifier | 0 | 0.362695| 
| RandomForestClassifier | 1 | 0.362672| 
| RandomForestClassifier | 2 | 0.362681| 
| RandomForestClassifier | 3 | 0.362681| 
| RandomForestClassifier | 4 | 0.362681| 
| LinearSVC | 0 | 0.420484| 
| LinearSVC | 1 | 0.422784| 
| LinearSVC | 2 | 0.418469| 
| LinearSVC | 3 | 0.421621| 
| LinearSVC | 4 | 0.424243| 
| LogisticRegression | 0 | 0.417378| 
| LogisticRegression | 1 | 0.416320| 
| LogisticRegression | 2 | 0.415823| 
| LogisticRegression | 3 | 0.414305| 
| LogisticRegression | 4 | 0.418032| 

Give this a plot:

![](/assets/img/2019-06-11-categorising-song-genre-by-analysing-lyrics/first.png)

LinearSVC and LogisticRegression perform better than RandomForest, with LinearSVC having a slight advantage with a median accuracy of around 42%.

I also did the same thing for word2vec-mean, word2vec-tfidf, word2vec-sif, glove-mean, glove-tfidf, glove-sif, fasttext-mean, fasttext-tfidf, and fasttext-sif.