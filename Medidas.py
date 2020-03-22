#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division
import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import nltk

tokenize = lambda doc: doc.lower().split(" ") #usar nltk

def tokenize(doc):
    doc = doc.lower()
    doc = nltk.word_tokenize(doc)
    return doc

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]
tokenized_documents = [tokenize(doc) for doc in all_documents]

def matching_similarity(query, document):
    intersection = set(query).intersection(set(document))
    return len(intersection)

#print 'matching similarity', matching_similarity(tokenized_documents[0], tokenized_documents[1])

def dice_similarity(query, document):
    #query = set(query)
    #document = set(document)
    intersection = set(query).intersection(document)
    difference = set(query).difference(set(document))
    return 2*len(intersection)/(len(query)+len(document))

#Problema con palabras repetidas!
print 'dice similarity', dice_similarity(tokenized_documents[2], tokenized_documents[2])

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

#print 'jaccard similarity', jaccard_similarity(document_0, document_1)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

#print 'term frequency', term_frequency('china', tokenize(document_0))

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

#print 'term frequency', sublinear_term_frequency('china', tokenize(document_0))

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

#print 'term frequency', augmented_term_frequency('china', tokenize(document_0))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

#tokenized_documents = [tokenize(d) for d in all_documents]
#print 'idf', inverse_document_frequencies(tokenized_documents)

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

tfidf_representation = tfidf(all_documents)
#print inverse_document_frequencies(tokenized_documents).keys()
#print len(tfidf_representation[0]), tfidf_representation[0] #arma un vector del tamanio del lexico total

def tfidf_onedoc(documents, doc_index):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    doc_tfidf = []
    for term in idf.keys():
        tf = sublinear_term_frequency(term, documents[doc_index])
        doc_tfidf.append(tf * idf[term])
        print term, tf * idf[term]
    return doc_tfidf

#tfidf_onedoc(all_documents, 0)

#in Scikit-Learn
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

#print 'cosine similarity my_tfidf', cosine_similarity(tfidf_representation[0], tfidf_representation[0])
#print 'cosine similarity sklearn', cosine_similarity(sklearn_representation.toarray()[0], sklearn_representation.toarray()[0])

def compare_models():
    #Compare our model with sklearn
    tfidf_representation = tfidf(all_documents)
    our_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(tfidf_representation):
        for count_1, doc_1 in enumerate(tfidf_representation):
            our_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    skl_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
        for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
            skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    for x in zip(sorted(our_tfidf_comparisons, reverse = True), sorted(skl_tfidf_comparisons, reverse = True)):
        print x

#compare_models()

def all_measures(query, document):
    tokenized_query = tokenize(query)
    tokenized_doc = tokenize(document)
    tfidf_representation = tfidf(all_documents)
    print 'matching similarity', matching_similarity(tokenized_query, tokenized_doc)
    print 'dice similarity', dice_similarity(tokenized_query, tokenized_doc)
    print 'jaccard similarity', jaccard_similarity(tokenized_query, tokenized_doc)
    #print 'cosine similarity', cosine_similarity(query, document) llamar a tfidf repre

all_measures(document_0, document_0)

def getKey(item):
    return item[0]
l = [[2, 3], [6, 7], [3, 34], [24, 64], [1, 43]]
sorted(l, key=getKey)


def get_all_similarities(tfidf_representation): #para usar con sklearn pasarle sklearn_representation.toarray()
    all = []
    l = [i for i in range (0,len(tfidf_representation))]
    combinations = list(itertools.combinations(l, 2))
    for i in combinations:
        cs = cosine_similarity(tfidf_representation[i[0]], tfidf_representation[i[1]])
        all.append((i[0], i[1], cs))
    all = [(c, a, b) for a, b, c in all] #put cosine similarity first in order to sort
    all = sorted(all, reverse=True)
    return all

#for i in get_all_similarities(tfidf_representation): print i

def get_max_similarities(tfidf_representation):
    all = get_all_similarities(tfidf_representation)
    return max(all)

#print get_max_similarities(tfidf_representation)

def run_over_a_corpus():
    corpus = []
    with open('300actas.txt') as f:
        for acta in f:
            corpus.append(acta)
    print 'repr'
    tfidf_representation = tfidf(corpus)
    print 'get'
    all = get_all_similarities(tfidf_representation)
    for i in all: print i
    print 'max similar', all[0]

#run_over_a_corpus()





