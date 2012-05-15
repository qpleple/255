#!/usr/bin/python

import os, sys
import math
import pickle
import re
from copy import deepcopy
from termcolor import cprint , colored
from stemming.porter2 import stem
from math import log, fabs, pow
from svmutil import *

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

red    = lambda x: cprint(x, 'white', 'on_red')
yellow = lambda x: cprint(x, 'yellow')
green  = lambda x: cprint(x, 'green')
blue   = lambda x: cprint(x, 'blue')

def store(obj, filename):
  with open('cache/' + filename + '.pickle', 'w') as f:
    pickle.dump(obj, f)
    
def load(filename):
  with open('cache/' + filename + '.pickle') as f:
    return pickle.load(f)
   
# corpus = {
#   'blah.txt': {
#     'label': 1
#     'words'  : ['how', 'are', 'you', '?', ...]
#   }
#   'aaa.txt': {...}
#   ...
# }
def readCorpus():
  corpus = {}
  directories = {'data/review_polarity/pos/': 1, 'data/review_polarity/neg/': 0}
  for (directory, label) in directories.items():
    _, _, filenames = os.walk(directory).next()
    for filename in filenames:
      with open(directory + filename) as f:
        corpus[filename] = {'label': label, 'words': f.read().split()}
  return corpus
   
# -----------------------------------------------------------------------------
# Bow
# -----------------------------------------------------------------------------

# stats = {
#   'blah': {
#     'tp' : 12
#     'fp' : 8
#     'fn' : 988  # fn = 1000 - tp
#     'tn' : 992  # tn = 1000 - fp
#     'idx': 42
#   }
#  'aaa': {...}
#  ...
# }

# bow = {
#   'blah.txt': {
#     'label': 1
#     'bag'  : {12:3, 34:5, 8532:1, ...}
#   }
#   'aaa.txt': {...}
#   ...
# }

def computeBow(corpus, vocab, lookupvocab):
  

  stats = {}
  bow   = {}
  for (filename, filedata) in corpus.items():
    # Stats
    for w in set(filedata['words']):
      if w not in stats:
        stats[w] = {'tp': 0, 'fp': 0}
      if filedata['label']:
        stats[w]['tp'] += 1
      else:
        stats[w]['fp'] += 1
    
    # Bow
    bag = {}
    for w in filedata['words']:
      i = lookupvocab[w]
      if i not in bag:
        bag[i] = 0
      bag[i] += 1
    bow[filename] = {'bag': bag, 'label': filedata['label']}
    
  # complete stats
  for w in stats:
    stats[w]['fn'] = 1000 - stats[w]['tp']
    stats[w]['tn'] = 1000 - stats[w]['fp']
    
  
  return stats, bow


# vocab, stats, bow = (load('vocab'), load('stats'), load('bow'))

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

def storeVocab():
  yellow('loading')
  corpus = readCorpus()

  yellow('computing')
  allWords = set()
  for (filename, filedata) in corpus.items():
    words = filedata['words']
    allWords.update(words)
    allWords.update(map(stem, words))

  yellow('storing')
  store(list(allWords), 'vocab')

def storeCorpus():
  store(readCorpus(), 'corpus')

def stemCorpus(corpus):
  for (filename, filedata) in corpus.items():
    for (i, w) in enumerate(filedata['words']):
      s = stem(w)
      if s != w:
        filedata['words'][i] = s

def storeStemmedCorpus():
  yellow('reading')
  corpus = load('corpus')
  yellow('stemming')
  stemCorpus(corpus)
  yellow('storing')
  store(corpus, 'corpus-stemmed')

def removeWordsFromCorpus(corpus, wordsToBeRemoved):
  for (filename, filedata) in corpus.items():
    filedata['words'] = [w for w in filedata['words'] if w not in wordsToBeRemoved]

def storePunctuation(corpus):
  allWords = set()
  for (filename, filedata) in corpus.items():
    allWords.update(filedata['words'])
  r = re.compile('[^0-9a-z]')
  punctuation = set([w for w in allWords if len(w) == 1 and r.match(w)])
  store(punctuation, 'punctuation')

def experiment():
  results = open('results.csv', 'w')
  options = {}
  yellow('loading')
  green('vocab')
  vocab        = load('vocab')
  lookupvocab = dict([(w, i) for (i, w) in enumerate(vocab)])
  green('corpus')
  corpusRaw    = load('corpus')
  green('corpus stemmed')
  corpuStemmed = load('corpus-stemmed')
  green('corpus stopwords')
  stopwords    = load('stopwords')
  green('corpus punctuation')
  punctuation  = load('punctuation')

  count = 1

  for doStemming in [True, False]:
    for featureTransformation in ['none', 'indic', 'logplusone', 'logodds']:
      for excludeUnitWords in [True, False]:
        for excludeStopWords in [True, False]:
          for excludePunctuation in [True, False]:
            yellow({
              'featureTransformation': featureTransformation,
              'doStemming': doStemming,
              'excludeStopWords': excludeStopWords,
              'excludePunctuation': excludePunctuation,
              'excludeUnitWords': excludeUnitWords,
            })
            params  = (featureTransformation, doStemming, excludeStopWords, excludePunctuation, excludeUnitWords)
            bowName = "bow.feat-%s.stem-%s.xSw-%s.xPun-%s.x1w-%s" % params
            yellow(bowName)

            # Stemming
            green('copy')
            if doStemming:
              corpus = deepcopy(corpuStemmed)
            else:
              corpus = deepcopy(corpusRaw)
            
            wordsToBeRemoved = set()
            # Stopwords
            if excludeStopWords:
              green('stopwords')
              wordsToBeRemoved.update(stopwords)
            # Punctuation
            if excludePunctuation:
              green('punctuation')
              wordsToBeRemoved.update(punctuation)
            green('remove words')
            removeWordsFromCorpus(corpus, wordsToBeRemoved)

            print " ".join(corpus['cv526_12083.txt']['words'])

            green('computing bow')
            stats, bow = computeBow(corpus, vocab, lookupvocab)
            if excludeUnitWords:
              green('list unit words')
              wordsToBeRemoved = set()
              for (w, m) in stats.items():
                if m['tp'] + m['fp'] == 1:
                  wordsToBeRemoved.add(w)
              green('removing those')
              removeWordsFromCorpus(corpus, wordsToBeRemoved)
              green('recomputing the')
              stats, bow = computeBow(corpus, vocab, lookupvocab)

            green('feature transformation: %s' % featureTransformation)
            transformBow(featureTransformation, bow, stats, lookupvocab)

            green('Libsvm format')
            labels   = [v['label'] for (k, v) in bow.items()]
            features = [v['bag']   for (k, v) in bow.items()]

            
            pb  = svm_problem(labels, features)
            for c in range(-2, 9):
              green('log10(C) = %i' % c)
              blue('%s' % ((count * 100) / 704))
              count += 1
              param = svm_parameter('-t 0 -c %s -v 5 -q' % pow(10, - c))
              green('Train...')
              acc = svm_train(pb, param)
              results.write('%s, %s, %s\n' % (', '.join(map(str, list(params))), c, acc))
  results.close()


def transformBow(featureTransformation, bow, stats, lookupvocab):
  if featureTransformation == 'none':
    return

  if featureTransformation == 'logodds':
    weights = {}
    for (w, m) in stats.items():
      i = lookupvocab[w]
      tp = max(0.5, float(m['tp']))
      fp = max(0.5, float(m['fp']))
      tn = max(0.5, float(m['tn']))
      fn = max(0.5, float(m['fn']))
      weights[i] = fabs(log(tp / fn) - log(fp / tn))

  for (filename, filedata) in bow.items():
    for (i, count) in filedata['bag'].items():
      if featureTransformation == 'indic':
        filedata['bag'][i] = 1
      elif featureTransformation == 'logplusone':
        filedata['bag'][i] = log(1 + count)
      elif featureTransformation == 'logodds':
        filedata['bag'][i] = weights[i]


# def dumpBowSvm(bow, name):
#   with open('data/%s.svm', name) as f:
#     for (filename, filedata) in bow.items():
#       for (i, count) in filedata['bag'].items():

def bla():
  labels = [1,1]
  features = [{1:1, 3:1}, {1:-1,3:-1}]
  pb  = svm_problem(labels, features)
  param = svm_parameter('-t 0 -c 4 -v 3 -q -b 1')
  model = svm_train(pb, param)
  print type(model)
