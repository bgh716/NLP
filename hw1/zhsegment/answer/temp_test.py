import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
#from ensegment import Pdist, datafile, product
from zhsegment import Pdist, datafile, product
from functools import reduce
from collections import defaultdict
from math import log10
import heapq



class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
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

        for i in chart:
            print(i,chart[i])

        ## Get the best segmentation ##
        r_index = len(text)-1
        segment_list = []
        while(r_index!=-1):
            r_entry = chart[r_index]
            segment_list.append(r_entry[1])
            r_index = r_entry[3]
        segment_list.reverse()
        
        return segment_list

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        #return product(self.Pw(w) for w in words)
        return sum(log10(self.Pw(w)) for w in words) 












#Pw = Pdist(data=datafile(os.path.join('data', 'count_1w.txt')))
Pw = Pdist(data1=datafile("data/count_1w.txt"), data2=datafile("data/count_2w.txt"))

segmenter = Segment(Pw)
print(len("新华社北京４月２３日电"))
li = segmenter.segment("新华社北京４月２３日电")


print(li)
