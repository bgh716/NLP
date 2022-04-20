import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
import heapq




class Segment:
    segment = None

    def __init__(self, Pw):
        self.Pw = Pw
        self.segment = self.segmentBigram


    def segmentBigram(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        
        # Baseline (see "Algorithm: Iterative segmenter" in Hw page)
        # Entry: (log-probability, word, start-position, back-pointer)
        chart = {-1:(0,"",-1,-2)}
        
        ## Initialize the heap ##
        h = []
        for word in self.Pw.c1:
            if(text.startswith(word)):
                    heapq.heappush(h, (log10(self.Pw.c2_probability(("<S>", word))), word, 0, -1))

        maxPrefixLen = 0
        prev_word_is_in_vocabulary = True

        while(maxPrefixLen < len(text)):
            if(not h):
                if(prev_word_is_in_vocabulary):
                    word = text[maxPrefixLen]
                    prevWord = chart[maxPrefixLen-1]
                    heapq.heappush(h, (log10(self.Pw.c1_probability(word))+prevWord[0], word, maxPrefixLen, maxPrefixLen-1))
                else:
                    word = text[maxPrefixLen]
                    prevWord = chart[maxPrefixLen-1]
                    heapq.heappush(h, (log10(self.Pw.c1_probability(word))+prevWord[0], prevWord[1]+word, prevWord[2], prevWord[3]))
                prev_word_is_in_vocabulary = False
                
            ## Iteratively fill in chart[i] for all i ##
            while(h):
                entry = heapq.heappop(h)
                endindex = entry[2]+len(entry[1])-1
                if(endindex in chart):
                    preventry = chart[endindex]
                    if(entry[0] > preventry[0]):
                        chart[endindex] = entry
                    else:
                        continue  ## we have already found a better segmentation of the prefix ending at 'endindex' ##
                else:
                    chart[endindex] = entry

                maxPrefixLen = max(maxPrefixLen, endindex + 1)
                
                if(endindex+1< len(text) and text[endindex+1] in "０１２３４５６７８９"):
                    i=endindex+1
                    while(i<len(text) and (text[i] in "·０１２３４５６７８９")):
                        i = i+1
                    
                    if(i<len(text) and text[i] in "日月年万亿"):
                        newword = text[endindex+1:i+1]
                    else:
                        newword = text[endindex+1:i]
                    newentry = (entry[0], newword, endindex+1, endindex)
                    if(newentry not in h):
                        prev_word_is_in_vocabulary = True
                        heapq.heappush(h, newentry)
                
                for newword in self.Pw.c1:
                    if(text.startswith(newword, endindex+1)):
                        newentry = (log10(self.Pw.c2_probability((entry[1],newword)))+entry[0], newword, endindex+1, endindex)
                        if(newentry not in h):
                            prev_word_is_in_vocabulary = True
                            heapq.heappush(h, newentry)

        ## Get the best segmentation ##
        r_index = len(text)-1
        segment_list = []
        while(r_index!=-1):
            r_entry = chart[r_index]
            segment_list.append(r_entry[1])
            r_index = r_entry[3]
        segment_list.reverse()
        
        return segment_list
        
#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    
    c1={} # counts of occurences of unigrams in unigram dataset
    c2_N={}	# c2_N[w0] = sum(occurences of bigram (w0,w')) for all w' in Vocabulary, ie. total count of all bigrams in which w0 is the first word
    c2={}	# c2[w0][w1] = count for bigram "w0 w1" 
    V = 0  # number of words
    
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data1=[], data2=[], N=None, missingfn=None):
        for key,count in data1:
            self.V = self.V + 1
            self.c1[key] = self.c1.get(key, 0) + int(count)
        for key, count in data2:
            (key1, key2) = key.split(' ')
            self.c2_N[key1] = self.c2_N.get(key1, 0) + int(count)
            if(key1 not in self.c2):
                self.c2[key1] = {}
            if(key2 not in self.c2[key1]):
                self.c2[key1][key2] = 0
            self.c2[key1][key2] = self.c2[key1][key2] + int(count)
        
        self.N = float(N or sum(self.c1.values()))
        self.missingfn = missingfn or (lambda k, N: 1./((N+self.V)**len(k)))
        
        
    def c1_probability(self, key): 
        if key in self.c1: return (self.c1[key]+1)/(self.N+self.V)  
        else: return self.missingfn(key, self.N)

    def c2_probability(self, key):
        (key1, key2) = key
        # calculate the Pr(key2|key1), ie. the probability that the word key2 will occur immediately after key1
        countKey1 = self.c2_N.get(key1, 0)
        countKey1Key2 = 0
        if(countKey1):  # if the first word in the key occured as the first word in at least one bigram in the dataset, could also use (key1 in self.c2) as a condition here
            countKey1Key2 = self.c2[key1].get(key2, 0)
        else:
            return self.c1_probability(key2)
        
        
        w1=0.3
        w2=0.4
        w3=0.3
        a = 1
        
        pc1 = self.c1_probability(key2)
        pc2 = (a + countKey1Key2)/(self.V*a + countKey1)
        #pc2 = (countKey1Key2)/(countKey1)
                
        # method 1: Add-one (Laplacian) Smoothing
        return w1*pc1 + w2*pc2 + w3/self.V

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data1=datafile(opts.counts1w), data2=datafile(opts.counts2w))
    segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
