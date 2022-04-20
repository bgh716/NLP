import os, sys, optparse
import tqdm
import pymagnitude
from scipy import spatial

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.topn = topn
        self.wvecs =pymagnitude.Magnitude(wvec_file)
        self.wvecs_dict = {}
        for key, vector in self.wvecs:
            self.wvecs_dict[key] = vector
        #non context words adopted form https://github.com/Mchristos/lexsub/blob/master/tools.py
        self.non_context = ["it's", "she's", 'were', 'because', 'this', 'couldn', 'then', 'how', 'd', 'doesn', 'down', 's', 'they', 'she', "needn't", 'wasn', 'haven', 'between', "wouldn't", 'the', 'ma', "wasn't", 'until', 'my', 'himself', "that'll", 'by', 'about', 'in', "aren't", "should've", 'why', 'nor', 'before', 'when', 'we', 'here', 'only', "couldn't", 'ain', 'no', 'your', 'will', 'own', 'his', "you'll", 'are', 'and', 'most', 'do', 'now', "isn't", 'having', 'on', 'her', 'theirs', 'under', 'with', 'to', "mightn't", 'while', 'its', 'be', 'll', 'don', 'over', 'again', 'their', 'won', 'too', 'during', 'shan', 'herself', 'has', 'or', 'from', 'ours', 'into', 'our', 'above', 'wouldn', 'you', 'of', 'so', 't', 'he', 'doing', 'as', 'i', 'can', 'shouldn', 'have', 'at', 'other', 'hasn', 'more', 'yourselves', 'y', 'yours', 'very', 'themselves', 'which', 'these', 'being', 'both', 'aren', 'did', 'than', 'needn', 'for', 'itself', "haven't", 'through', 'weren', 'but', 'once', 'isn',  'ourselves', 'didn', 'not', 'yourself', 'mightn', 'after', 've', 'him', 'whom', "hasn't", 'a', 'hadn', "shouldn't", "mustn't", 'those', 'off',  'each', 'was', "didn't", "you'd", 'where', 'o', 'further', 'below', "shan't",  'myself', 'mustn', 'is', 'been', 'just', 'any', 'out', 'that', 'm', 'such',  'me', 'same', 'hers', 'some', 'had', 'does', 'against', 'should', "you've",  "doesn't", "you're", 'them', 'am', 'if', 'who', 'few', 'what', 'there',  "don't", "weren't", "won't", 'an', 'all', 're', 'it', 'up', "hadn't", "'ll", ',', '.']

    def substitutes(self, index, sentence, oldvec):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return (list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn)))) ##no context return
        #context return
        #candidates = (list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))
        #top10 = self.find_context(candidates, index, sentence, oldvec)
        #return list(map(lambda k: k, top10))
        
    def find_context(self, candidates, index, sentence, oldvec):
        candidates_dict = {}
        v = self.wvecs_dict[sentence[index]]
        context_words = []
        sentence_range = 1 ##around words from index
        for word in sentence[index-sentence_range:index]:
            if word in oldvec.wvecs_dict and word.isalpha() and word not in self.non_context:
                context_words.append(oldvec.wvecs_dict[word])
        for word in sentence[index+1:index+sentence_range+1]:
            if word in oldvec.wvecs_dict and word.isalpha() and word not in self.non_context:
                context_words.append(oldvec.wvecs_dict[word])
                
        bal = len(context_words)
        ##bal-add calculation
        if context_words:
            for word in candidates:
                candidates_dict[word] = ((1-spatial.distance.cosine(self.wvecs_dict[word], v)) * bal + sum((1-spatial.distance.cosine(oldvec.wvecs_dict[word], c)) for c in context_words)) / (bal*2)
        else:
            return candidates[:10]
            
        return list(map(lambda k: k[0],sorted(candidates_dict.items(), key=lambda x: x[1], reverse = True)[:10]))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-r", "--wordvecretrofile", dest="wordvecretrofile", default=os.path.join('data', 'glove.6B.100d.retrofit.magnitude'), help="retro word vectors file")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub_old = LexSub(opts.wordvecfile, int(opts.topn))
    lexsub = LexSub(opts.wordvecretrofile, int(opts.topn)*1) ## 10*n candidates
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split(), lexsub_old)))
