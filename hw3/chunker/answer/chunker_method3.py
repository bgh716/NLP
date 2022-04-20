# Code adapted from original code by Robert Guthrie

import os, sys, optparse, gzip, re, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import string
import random

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

def prepare_sequence(seq, to_ix, unk):
    idxs = []
    if unk not in to_ix:
        idxs = [to_ix[w] for w in seq]
    else:
        idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
    return torch.tensor(idxs, dtype=torch.long)

def noisy(seq):
    noisySeq = []
    for word in seq:
        if(len(word > 2)):
            continue

def prepareSemiCharVectors(seq, char_to_ix):
    n = len(string.printable)
    vectors = []
    for word in seq:
        #initialize subvectors
        b = torch.zeros(n)
        i = torch.zeros(n)
        e = torch.zeros(n)

        #encode word
        if(len(word)):
            b[char_to_ix[word[0]]] += 1
            if(len(word) > 2):
                for char in word[1:-1]:
                    i[char_to_ix[char]] +=1
                e[char_to_ix[word[-1]]] += 1
        
        #push vector
        newVector = torch.cat((b, i, e), dim=-1)
        vectors.append(newVector)
    vectors = torch.stack(vectors)
    return torch.tensor(vectors, dtype=torch.long)


class scRNNModel(nn.Module):
    def __init__(self, target_dim):
        torch.manual_seed(1)
        super(scRNNModel, self).__init__()
        self.hidden_dim = 128
        self.input_dim = 300
        self.target_dim = target_dim
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=False)
        self.hidden2word = nn.Linear(self.hidden_dim, self.target_dim)

    def forward(self, semicharVector):
        semicharVector = semicharVector.float()
        n = semicharVector.shape[0]
        lstm_out, _ = self.lstm(semicharVector.view(n, 1, -1))
        word_space = self.hidden2word(lstm_out.view(n, -1))
        word_scores = F.log_softmax(word_space, dim=1)
        return word_scores

class LSTMTaggerModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        torch.manual_seed(1)
        super(LSTMTaggerModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class LSTMTagger:

    def __init__(self, trainfile, modelfile, modelfile2, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64, clr=400):
        self.unk = unk
        self.embedding_dim = embedding_dim
        self.clr = clr
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.modelfile = modelfile
        self.modelfile2 = modelfile2
        self.modelsuffix = modelsuffix
        self.training_data = []
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        self.word_to_ix = {} # replaces words with an index (one-hot vector)
        self.tag_to_ix = {} # replace output labels / tags with an index
        self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag
        self.ix_to_word = {} # inverse of self.word_to_ix
        self.char_to_ix = {} # maps characters to integers

        i = 0
        for char in string.printable:
            self.char_to_ix[char] = i
            i+=1

        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.ix_to_word[len(self.word_to_ix)] = word
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        logging.info("word_to_ix:", self.word_to_ix)
        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)

        self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        self.scRNN = scRNNModel(target_dim=len(self.word_to_ix))
        self.scRNNOptimizer = optim.SGD(self.scRNN.parameters(), lr=0.02)

    def argmax(self, seq):
        unkIndex = self.word_to_ix[self.unk]
        output = []
        with torch.no_grad():
            raw_inputs = prepare_sequence(seq, self.word_to_ix, self.unk) 
            scRNN_inputs = prepareSemiCharVectors(seq, self.char_to_ix)
            tagger_inputs = torch.argmax(self.scRNN(scRNN_inputs), 1)
            tag_scores = self.model(tagger_inputs)
            for i in range(len(raw_inputs))
            # print(seq)
            # print(" ".join([self.ix_to_word[int(i)] for i in tagger_inputs]))
            # quit()
            for i in range(len(tagger_inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output

    def train_scRNN(self):
        loss_function = nn.CrossEntropyLoss()

        self.scRNN.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            for sentence, tags in tqdm.tqdm(self.training_data):
                self.model.zero_grad()

                reference = prepare_sequence(sentence, self.word_to_ix, self.unk) #issue
                noisyInput = prepareSemiCharVectors(sentence, self.char_to_ix)

                scores = self.scRNN(noisyInput)

                print(" ".join(list(sentence)))
                torch.argmax
                print(" ".join([self.ix_to_word[int(i)] for i in list(prediction)]))
                quit()

                loss = loss_function(prediction, reference)
                loss.backward()
                self.scRNNOptimizer.step()
            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile2 + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.scRNN.state_dict(),
                        'optimizer_state_dict': self.scRNNOptimizer.state_dict(),
                        'loss': loss
                    }, savefile)


    def train(self):
        loss_function = nn.NLLLoss()

        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            for sentence, tags in tqdm.tqdm(self.training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                tagger_input = prepare_sequence(sentence, self.word_to_ix, self.unk)
                targets = prepare_sequence(tags, self.tag_to_ix, self.unk)

                # Step 3. Run our forward pass.
                tag_scores = self.model(tagger_input)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                self.optimizer.step()

            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        loss = saved_model['loss']
        self.unk = saved_model['unk']
        self.word_to_ix = saved_model['word_to_ix']
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        self.model.eval()
        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax(sent))
        return decoder_output

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
    optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'chunker'), help="filename without suffix for model files")
    optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
    optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
    optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    modelfile2 = os.path.join("data", "scRNN")
    if opts.modelfile[-4:] == '.tar':
        modelfile = opts.modelfile[:-4]
    chunker = LSTMTagger(opts.trainfile, modelfile, modelfile2, opts.modelsuffix, opts.unk)
    # use the model file if available and opts.force is False

    # if not os.path.isfile(modelfile2 + opts.modelsuffix) or opts.force:
    chunker.train_scRNN()
    # if not os.path.isfile(opts.modelfile + opts.modelsuffix) or opts.force:
    #     print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
    #     chunker.train()
    decoder_output = chunker.decode(opts.inputfile)

    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
