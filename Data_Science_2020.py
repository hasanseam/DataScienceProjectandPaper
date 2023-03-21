# -*- coding: utf-8 -*-

# -- Sheet 3 --

# # The Hitchhiker's Guide to Happiness


# ## Abstract


# Using the “Happy moments” dataset we will train multiple classifiers predicting what kind of moments made a person happy, given their demographic information, and classifying descriptions of happy moments using NLP. We expect to have reliable models for these tasks at the end of our analysis and to be able to give more (scientific) insight into what makes different kinds of people happy.


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

import subprocess
#%%
print(subprocess.getoutput("python3 -m spacy download en"))

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('stopwords')
nltk.download('wordnet')

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

happy_moments_data = pd.read_csv("cleaned_hm.csv")
happy_moments_data

happy_moments_data.ground_truth_category.unique()

import string

# punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))

# Convert to list
#data = df.content.values.tolist()
data = happy_moments_data.cleaned_hm.values.tolist()

# Remove punctuation
# data = [punctuation_regex.sub('', x) for x in data]

# Remove new line characters
# data = [re.sub(r'\s+', ' ', sent) for sent in data]

pprint(data[:1])

stop_list = gensim.parsing.preprocessing.STOPWORDS

domain_specific_stop_words = ['happy', 'day', 'got', 'went', 'today', 
 'made', 'one', 'two', 'time', 'last', 'first', 'going', 'getting', 'took', 'found', 
 'lot', 'really', 'saw', 'see', 'month', 'week', 'day', 'yesterday', 'year', 'ago', 
 'now', 'still', 'since', 'something', 'great', 'good', 'long', 'thing', 'toi', 'without', 
 'yesteri', '2s', 'toand', 'ing']

domain_specific_stop_words.extend(['happiest', "new", "moment", 
        #"life", "like",
    ]
)

stop_list = stop_list.union(domain_specific_stop_words)

stop_list

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    # return stemmer.stem(WordNetLemmatizer().lemmatize(text))
    return WordNetLemmatizer().lemmatize(text)
    

# stop_list = [lemmatize_stemming(x) for x in stop_list]

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if len(token) >= 2 and token not in stop_list:
            result.append(lemmatize_stemming(token))
            
    return result

processed_data = []

for x in data:
    processed_data.append(preprocess(x))

processed_data

from gensim.models.phrases import Phrases, Phraser

phrases = Phrases(processed_data, min_count=1, threshold=1)

bigram = Phraser(phrases)
print(bigram[processed_data[0]])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(processed_data, min_count=2, threshold=1) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[processed_data], threshold=1)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[processed_data[0]]])

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

bigrammed_moments = make_bigrams(processed_data)

trigrammed_moments = make_trigrams(processed_data)
trigrammed_moments

processed_data

# Create Corpus
texts = processed_data
# texts = bigrammed_moments
#texts = trigrammed_moments

# Create Dictionary
id2word = corpora.Dictionary(texts)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Build LDA model

def build_lda_model(id2word, corpus, texts, num_topics):
    return gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=42,
                                           alpha='auto',
                                           per_word_topics=True
                                           )

# lda_model = build_lda_model(id2word, corpus, texts, 6)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = build_lda_model(id2word, corpus, texts, num_topics)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_value = coherencemodel.get_coherence()

        print(f'Model for {num_topics} topics: Coherence Score of {coherence_value}')

        coherence_values.append(coherence_value)

    return model_list, coherence_values

limit=10; start=2; step=1;

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=start, limit=limit, step=step)

chosen_model = model_list[6]

import matplotlib.pyplot as plt

# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.savefig('coherence_score.png', dpi=400)
plt.show()

# Visualize the topics 
pyLDAvis.enable_notebook() 
vis = pyLDAvis.gensim.prepare(chosen_model, corpus, id2word)
vis

# inputSentence = 'Today I submitted my final project presentation'
# inputSentence = 'I got promoted and now have a new job'
# inputSentence = 'I played a videogame with a friend'
# inputSentence = 'I spent some quality time with my girlfriend'
# inputSentence = 'I bought a new car'
# inputSentence = 'I\'ve gone on a world trip and traveled to America'
# inputSentence = 'I drove my daughter to school'
# inputSentence = 'I went on vacation with my family'
inputSentence = 'I went on vacation with my wife'
# inputSentence = 'I had dinner at a fancy restaurant'
inputSentence = preprocess(inputSentence)

print(inputSentence)

doc_vector = id2word.doc2bow(inputSentence)
print(doc_vector)
doc_topics = chosen_model[doc_vector]

print(data[doc_topics[2][0][0]])
# print(data[doc_topics[2][1][0]])
# print(data[doc_topics[2][2][0]])
# print(data[doc_topics[2][3][0]])

doc_topics

# Print the Keyword in the 10 topics
pprint(chosen_model.print_topics())
doc_lda = chosen_model[corpus]

from gensim.similarities import Similarity
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile

index_tmpfile = get_tmpfile("index")
index = Similarity(index_tmpfile, corpus, num_features=len(id2word))



sims = index[id2word.doc2bow(inputSentence)]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[:3])

print(data[sims[0][0]])
print(data[sims[1][0]])
print(data[sims[2][0]])

