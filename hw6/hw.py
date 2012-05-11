#!/usr/bin/python

import os, sys
import math
import pickle
from termcolor import colored

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

red    = lambda x: colored(x, 'red')
yellow = lambda x: colored(x, 'yellow')
green  = lambda x: colored(x, 'green')
blue   = lambda x: colored(x, 'blue')

def store(obj, filename):
  with open('cache/' + filename + '.pickle', 'w') as f:
    pickle.dump(obj, f)
    
def load(filename):
  with open('cache/' + filename + '.pickle') as f:
    return pickle.load(f)
    
# -----------------------------------------------------------------------------
# Bow
# -----------------------------------------------------------------------------

def computeBow():
  bow = computeBowFrom('data/review_polarity/pos/', 1)
  bow = computeBowFrom('data/review_polarity/pos/', 0, bow)
  return bow
  
def computeBowFrom(directory, label, bow = {}):
  pass
