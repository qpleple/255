#!/usr/bin/python

import os, sys
import math
import pickle
import re
from copy import deepcopy
from termcolor import cprint , colored
from stemming.porter2 import stem

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

def computeBow(corpus, vocab):
  lookupvocab = dict([(w, i) for (i, w) in enumerate(vocab)])

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
    for k in stats[w]:
      if k != 'idx':
        stats[w][k] = max(0.5, float(stats[w][k]))
  
  return stats, bow
  
def testBow(vocab, stats, bow):
  for filename in bow:
    # Set of words from the file
    label = bow[filename]['label']
    if label:
      directory = 'data/review_polarity/pos/'
    else:
      directory = 'data/review_polarity/neg/'
    with open(directory + filename) as f:
      words = set(f.read().split())
    # Set of words from the bow
    wordsbow = set([vocab[i] for i in bow[filename]['bag']])
    if wordsbow != words:
      red('error for ' + filename)
      yellow('Set of words from the file')
      print(words)
      yellow('Set of words from the bow')
      print(wordsbow)
      return
  green('passed')
  

def storeBow():
  yellow('reading')
  corpus = readCorpus()
  yellow('computing')
  vocab, stats, bow = computeBow(corpus)
  yellow('testing')
  testBow(vocab, stats, bow)
  yellow('storing')  
  store(vocab, 'vocab')
  store(stats, 'stats')
  store(bow, 'bow')

# vocab, stats, bow = (load('vocab'), load('stats'), load('bow'))

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

# def storeAllCorpus():
#   yellow('reading')
#   rawCorpus = readCorpus()
#   yellow('building')
#   corpuses = buildAllCorpuses(rawCorpus)
#   yellow('storing')
#   store(corpuses, 'corpuses')

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

def buildAllCorpuses():
  options = {}
  yellow('loading')
  green('vocab')
  vocab        = load('vocab')
  green('corpus')
  corpusRaw    = load('corpus')
  green('corpus stemmed')
  corpuStemmed = load('corpus-stemmed')
  green('corpus stopwords')
  stopwords    = load('stopwords')
  green('corpus punctuation')
  punctuation  = load('punctuation')

  for doStemming in [True, False]:
    #for featureTransformation in ['none', 'indic', 'logplusone', 'logodds']:
      for excludeUnitWords in [True, False]:
        for excludeStopWords in [True, False]:
          for excludePunctuation in [True, False]:
            yellow({
              'doStemming': doStemming,
              'excludeStopWords': excludeStopWords,
              'excludePunctuation': excludePunctuation,
              'excludeUnitWords': excludeUnitWords,
            })

            #  "doStemming:%s, excludeStopWords:%s, excludePunctuation:%s"
            #% (doStemming, excludeStopWords, excludePunctuation))
            
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

            stats, bow = computeBow(corpus, vocab)
            print bow['cv526_12083.txt']


            

def buildDataSet(vocab, stats, bow, options):
  wordsToRemove = set()

  # --- Punctuation ---
  if options['excludePunctuation']:
    r = re.compile('[^0-9a-z]')
    punctuation = set([x for x in vocab if len(x) == 1 and r.match(x)])
    wordsToRemove.update(punctuation)
  
  # --- Stopwords ---
  if options['excludeStopWords']:
    wordsToRemove.update(load('stopwords'))
    
  if options['doStemming']:
    for w in stats:
      stemmed = stem(w)
      if stemmed != w:
        stats[w]['stemmed'] = stemmed
      stats[stemmed]['tp'] += stats[w]['tp']
      stats[stemmed]['fp'] += stats[w]['fp']
      stats[stemmed]['fn'] = 1000 - stats[stemmed]['tp']
      stats[stemmed]['tn'] = 1000 - stats[stemmed]['fp']
    # for (filename, filedata) in bow.items():
    #   for w in filedata['bag']
      