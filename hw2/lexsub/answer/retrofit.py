import pymagnitude
import re
import math
import os.path
from numpy import array
from functools import reduce

def retrofitWordVectors(n, wvecFilepath, outputFilepath):
    oldWordVectors = pymagnitude.Magnitude(wvecFilepath)
    wordVectors = loadWordVectors(oldWordVectors)
    
    synonymSets = loadSynonyms(os.path.join(my_path, "../data/lexicons/wordnet-synonyms.txt"), {})
    #synonymSets = loadSynonyms(os.path.join(my_path, "../data/lexicons/wordnet-synonyms+.txt"), {})
    #synonymSets = loadSynonyms(os.path.join(my_path, "../data/lexicons/ppdb-xl.txt"), {})
    #synonymSets = loadSynonyms(os.path.join(my_path, "../data/lexicons/framenet.txt"), {})
    
    
    #weights. alpha for current word vector. beta for synonyms.
    alpha = 1
    beta = 1
    
    # simplest method adapted from https://github.com/mfaruqui/retrofitting/blob/master/retrofit.py
    for i in range(n):
        print(f"{i}-th sweep:", end='')
        newWordVectors = {}
        for word in wordVectors:    # for each word vector
            if word not in synonymSets:
                newWordVectors[word] = oldWordVectors.query(word)
                continue
        
            oldWordVector = oldWordVectors.query(word)
            vectorSum = reduce(lambda x,y:x+y, (wordVectors[synonym]*beta for synonym in synonymSets[word] if synonym in wordVectors), array([0]*oldWordVector.ndim))
            synonymCount = sum(1 for synonym in synonymSets[word] if synonym in wordVectors)
            
            if synonymCount > 0:
                try:
                    alpha = beta*synonymCount * 0.5
                    vectorSum += oldWordVector * (alpha) 
                    
                    #vectorSum += synonymCount*oldWordVector  #assumes a_i weights are all 1
                    #vectorSum = vectorSum/(2*synonymCount)  #assumes b_i,j weights are all 1
                    vectorSum = vectorSum/(synonymCount*beta+alpha)
                except:
                    print(word, synonym, vectorSum, oldWordVector, synonymCount)
            newWordVectors[word] = vectorSum
        wordVectors = newWordVectors
        print("finished")

    # setMembership = {}
    # for key, members in synonymSets:
    #     if(key in setMembership):
    #         setMembership[key] = [key]
    #     else:
    #         setMembership[key].append(key)
    #     for word in members:
    #         if(word in setMembership):
    #             setMembership[word] = [key]
    #         else:
    #             setMembership[word].append(key)

    # for i in range(n):
    #     for word in wordVectors:
    #         synonymCount = 0
    #         vectorSum = self.wvecs[word]    # original word vector used here in accordance with equation provided in assignment 
    #         for setKey in setMembership[word]:
    #             synonymSet = synonymSets[setKey]
    #             for synonym in synonymSet:
    #                 if(synonym != word):
    #                     vectorSum += wordVectors[synonym]
    #                     synonymCount += 1
    #         vectorSum += synonymCount*self.wvecs[word]
    #         wordVectors[word] = vectorSum/(2*synonymCount)

    sep = " "
    delim = "\n"
    with open(outputFilepath, "w") as f:
        for word in wordVectors:
            outputString = word + sep + sep.join([str(i) for i in wordVectors[word]]) + delim
            f.write(outputString)
    return


    
def loadSynonyms(filepath, synonymSets):
    sep = " "
    
    with open(filepath) as fh:
        for line in fh:
            synonyms = line.lower().strip().split(sep)
            
            synonyms = [filterWords(i) for i in synonyms]
          
            ############## using full graph
            #for i in range(0, len(synonyms)):
            #    if (synonyms[i] not in synonymSets):
            #        synonymSets[synonyms[i]] = set()
            #    synonymSets[synonyms[i]].update(synonyms[:i]+synonyms[i+1:])
            
            ############## using synonym graph
            if (synonyms[0] not in synonymSets):
                synonymSets[synonyms[0]] = set()
            synonymSets[synonyms[0]].update(synonyms[1:])
    return synonymSets

def loadWordVectors(readOnlyWordVectors):
    wordVectors = {}
    vectorDegree = readOnlyWordVectors.dim
    epsilon = 1e-6
    for key, vector in readOnlyWordVectors:
        wordVectors[key] = vector
        #wordVectors[key] /= math.sqrt((wordVectors[key]**2).sum() + epsilon)    # bring each component of the vector into the range [0, 1], should prevent overflows during retrofitting
        
    return wordVectors

#below function adapted from https://github.com/mfaruqui/retrofitting/blob/master/retrofit.py
numberRegex = re.compile(r'\d+.*') # matches digits followed by any sequence of characters, or nothing
def filterWords(word):
    word = word.lower()
    if numberRegex.search(word):    # word contains numbers
        return "---num---"
    if re.sub(r"\W+", "", word) == "":  # word contains only punctuation
        return "---punc---"
    return word

my_path = os.path.abspath(os.path.dirname(__file__)) ##change relative file paths here
wvecFilepath = os.path.join(my_path, "../data/glove.6B.100d.magnitude")
outputFilepath = os.path.join(my_path, "../data/glove.6B.100d.retrofit.txt")

T = 25 ##change iteration parameter T here




retrofitWordVectors(T, wvecFilepath, outputFilepath)

#python3 -m pymagnitude.converter -i data/glove.6B.100d.retrofit.txt -o data/glove.6B.100d.retrofit.magnitude



