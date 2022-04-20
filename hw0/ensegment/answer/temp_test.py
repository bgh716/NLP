import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
#from ensegment import Pdist, datafile, product
from default import Pdist, datafile, product
from functools import reduce
from collections import defaultdict
from math import log10


def memo(f):
    "Memoize function f."
    table = {}
    p=len(table)==0
    def fmemo(self, args, p=False):
        if args not in table:
            table[args] = f(self, args,p)
        return table[args]
    fmemo.memo = table
    return fmemo

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    @memo
    def segment(self, text, p=False):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        candidates = ([first]+self.segment(rem) for first,rem in self.splits(text))
        

        if(p):
            temp = list([[first]+self.segment(rem) for first,rem in self.splits(text)])
            temp.sort(key=self.Pwords, reverse=True)
            for i in range(min(5,len(temp))):
                print(temp[i])
                print(self.Pwords(temp[i]))
        
        return max(candidates, key=self.Pwords)

    def splits(self, text, L=20):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:]) 
                for i in range(min(len(text), L))]

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        #return product(self.Pw(w) for w in words)
        return sum(log10(self.Pw(w)) for w in words) 












#Pw = Pdist(data=datafile(os.path.join('data', 'count_1w.txt')))
Pw = Pdist(data=datafile("../data/count_1w.txt"))

segmenter = Segment(Pw)

#li = segmenter.segment("unclimatechangebody", p=True)
#li = segmenter.segment("gerger", p=True)
li = segmenter.segment("gergerger", p=True)


print(li)
