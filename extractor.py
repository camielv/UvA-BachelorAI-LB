# Extractor extracts a column from a csv file
import csv
import nltk
import perceptron
import random

class Main():

    # Open a file
    file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')     

    # used dictionaries, with enlarged scope..?
    sentence = {}
    sentiment = {}
    corpus = {}
    probWord = {}
    probSent = {}
    p = perceptron.Perceptron()    
    trainSet = []
    testSet = []
    validationSet = []
    distribution = (0.7, 0.1, 0.2) # train, test, validate

    def __init__(self):
        self.makeCorpus( )
        self.calcProbability( )
        self.trainPerceptron( )
        self.printResults( )
        
    def makeCorpus(self):
        print 'Creating corpus...'
        self.sentence = {}
        self.sentiment = {}

        # Initialize counter
        i = 0

        # Collect sentences and sentiments
        for entry in self.file1:
            # Do not include header
            if i == 0:
                i+=1
                continue
            
            # The actual message is the 9th attribute, sentiment is the 4th
            self.sentence[i - 1] = entry[9]
            self.sentiment[i - 1] = float(entry[4])

            # Assign at random to train, test or validation set
            r = random.random()
            if ( r < self.distribution[0] ):
                self.trainSet.append(i-1)
            else:
                self.testSet.append(i-1)
            
            # Stop at 10000
            i += 1
            if ( i == 10000 ): break

        number_of_items = i - 1; # -1 because of header  (== len(sentence))

        # Create corpus and count word frequencies
        self.corpus = {}
        
        for j in self.trainSet:
            # Tokenize the sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[j] )

            # Check for sentiment 
            sent = self.sentiment[j]

            # Iterate over every token
            for token in tk_sentence:
                if token in self.corpus:
                    if sent != 0:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1] + 1
                    else:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1]
                else:
                    if sent != 0:
                        self.corpus[token] = 1, 1
                    else:
                        self.corpus[token] = 1, 0

        print 'Calculating unigram probability.'
        # Corpus created, calculate words probability of sentiment based on frequency
        self.probWord = {}
        for token in self.corpus.keys():
            self.probWord[token] = float(self.corpus[token][1]) / self.corpus[token][0]
            # print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'

    def calcProbability(self):
        # Probability of sentiment per word calculated, estimate sentence probability of sentiment
        self.probSent = {}

        for i in self.trainSet:
                p = 1
                tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
                for token in tk_sent:
                    p = p + self.probWord[token]
                self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
                #print i, 'PROB', self.probSent[i], 'SENT', self.sentiment[i]
        
        
    def trainPerceptron(self):
        print 'Training perceptron.'
        spsv = self.probSent
        ssv  = [x != 0 for x in self.sentiment.values()]
                
        trainingSet = dict()
        for i in self.trainSet:
            trainingSet[(spsv[i],)] = ssv[i]
                 
        self.p.train(trainingSet)
        print 'Found threshold: ', self.p.threshold / self.p.weights[0]

        print 'Validating perceptron.'

        # Calculate probability for test sentences
        for j in self.testSet:
            p = 1
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[j] )
            for token in tk_sent:
                try:
                    p = p + self.probWord[token]
                except:
                    # if word does not occur in corpus, ignore (can try katz backoff later?)
                    pass
            # store the probability in  dictionary     
            self.probSent[j] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
            #print i, 'PROB', self.probSent[i], 'SENT', self.sentiment[i]

    def printResults(self):
        t = self.p.threshold / self.p.weights[0]
        confusion = {}
        confusion["tp"] = 0
        confusion["tn"] = 0
        confusion["fp"] = 0
        confusion["fn"] = 0
        for i in self.testSet:
                if self.probSent[i] > t:
                    if self.sentiment[i] == 0:
                        confusion["fp"] += 1
                    else:
                        confusion["tp"] += 1
                if self.probSent[i] < t:
                    if self.sentiment[i] == 0:
                        confusion["tn"] += 1
                    else:
                        confusion["fn"] += 1
        print 'Results for test set: '
        print confusion
        print 'accuracy = ', float(confusion["tp"] + confusion["tn"]) / (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
        print 'precision = ', float(confusion["tp"]) / (confusion["tp"] + confusion["fp"] )

m = Main()
