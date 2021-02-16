# 09 - 02 - 2021
# Pranav Pathare

#--------------------------------------------------------------------------------------------------------
# Imports

from flask import Flask, request, jsonify
import re

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from gensim.models import word2vec
import gensim.downloader as api

import pandas as pd
import numpy as np

import sklearn
from sklearn.metrics.pairwise import cosine_similarity

#--------------------------------------------------------------------------------------------------------
# Preprocessing Functions

def clean_sentence(sentence, stopwords=False):
    """ Cleans Stopwords from a sentence"""
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9/s]', ' ', sentence)

    if stopwords:
        sentence = remove_stopwords(sentence)

    return sentence

def get_cleaned_sentences(df, stopwords=False):
    """Fetches cleaned sentences from the data"""
    sentences = df[['Complaint Text']]
    cleaned_sentences = []

    for index, row in df.iterrows():
        cleaned = clean_sentence(row['Complaint Text'], stopwords)
        cleaned_sentences.append(cleaned)

    return cleaned_sentences

def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf, sentences):
    """ Computes Cosine simiarities and returns sentence with best match"""
    max_sim = -1
    index_sim = -1
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        if sim > max_sim:
            max_sim = sim
            index_sim = index

    return FAQdf.iloc[index_sim, 1]


def getWordVec(word, model):
    """ Loads the model """
    samp = model['computer']
    vec = [0] * len(samp)
    try:
        vec = model[word]
    except:
        vec = [0] * len(samp)

    return vec


def getPhraseEmbedding(phrase, embeddingmodel):
    """ Fetches embedding for input phrase"""
    samp = getWordVec('computer', embeddingmodel)
    vec = np.array([0] * len(samp))
    den = 0
    for word in phrase.split():
        den = den + 1
        vec = vec + np.array(getWordVec(word, embeddingmodel))

    return vec.reshape(1, -1)
#--------------------------------------------------------------------------------------------------------

filename = 'sample_data_complaint.xlsx'
df = pd.read_excel(filename, usecols=["Complaint Text", "Resolution"])
df.columns = ["Complaint Text", "Resolution"]

# glove_model = gensim.models.KeyedVectors.load("./glovemodel.mod")
# v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
v2w_model = gensim.models.KeyedVectors.load("./glovemodel.mod")

#--------------------------------------------------------------------------------------------------------

cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)

sentences = cleaned_sentences_with_stopwords
sentence_words = [[word for word in document.split()] for document in sentences]

dictionary = corpora.Dictionary(sentence_words)

bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]

sent_embeddings = []
for sent in cleaned_sentences:
    sent_embeddings.append(getPhraseEmbedding(sent, v2w_model))

#--------------------------------------------------------------------------------------------------------
# App

app = Flask(__name__)

@app.route('/predict_solution', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        return "Hello"

    content = request.get_json()
    problem_text = content["problem_text"]

    if problem_text.strip() == "":
        return jsonify({'solution_text': " "})

    question = clean_sentence(problem_text, stopwords = False)
    question_embedding = dictionary.doc2bow(question.split())
    question_embedding = getPhraseEmbedding(question, v2w_model)

    result = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, df, cleaned_sentences)

    return jsonify( {'solution_text': result } )
