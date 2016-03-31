# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import codecs

from .. import normal
from .. import seg
from ..classification.bayes import Bayes
import pymongo

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'sentiment.marshal')
conn = pymongo.Connection("127.0.0.1",27017)
db = conn.testDB

class Sentiment(object):

    def __init__(self):
        self.classifier = Bayes()

    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip)

    def load(self, fname=data_path, iszip=True):
        self.classifier.load(fname, iszip)

    def handle(self, doc):
        words = seg.seg(doc)
        words = normal.filter_stop(words)
        return words

    def train(self, neg_docs, pos_docs):
        data = []
        for sent in neg_docs:
            data.append([self.handle(sent), 'neg'])
        for sent in pos_docs:
            data.append([self.handle(sent), 'pos'])
        self.classifier.train(data)

    def classify(self, sent):
        ret, prob = self.classifier.classify(self.handle(sent))
        if ret == 'pos':
            return prob
        return 1-prob


classifier = Sentiment()
classifier.load()

#add datasource from mongodb
def trainFromMongo(positiveCollection, negativeCollection):
    neg_docs = db.negativeCommentsFixSchema.find()["CommentContent"]
    pos_docs = db.positiveCommentsFixSchema.find()["CommentContent"]
    train(neg_docs, pos_docs)

#add datasource from file
def trainFromFile(neg_file, pos_file):
    neg_docs = codecs.open(neg_file, 'r', 'utf-8').readlines()
    pos_docs = codecs.open(pos_file, 'r', 'utf-8').readlines()
    train(neg_docs, pos_docs)

def train(negs, poss):
    global classifier
    classifier = Sentiment()
    classifier.train(negs, poss)


def save(fname, iszip=True):
    classifier.save(fname, iszip)


def load(fname, iszip=True):
    classifier.load(fname, iszip)


def classify(sent):
    return classifier.classify(sent)
