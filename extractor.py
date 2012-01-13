# Extractor extracts a column from a csv file
import csv
import nltk
import perceptron

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

    def __init__(self):
        self.makeCorpus()
        self.calcProbability()
        self.trainPerceptron()
        self.printResults()
        
    def makeCorpus(self):
        print 'Creating corpus...'
        self.sentence = {}
        self.sentiment = {}

        # do not include header!
        i = 0

        # Collect sentences and sentiments
        for entry in self.file1:
            if i == 0:
                i+=1
                continue
            
            # The actual message is the 9th attribute, sentiment is the 4th
            self.sentence[i - 1] = entry[9]
            self.sentiment[i - 1] = float(entry[4])
            
            # Stop at 20 (later remove this)
            i += 1
            if ( i == 9999 ): break

        number_of_items = i - 1; # -1 because of header  (== len(sentence))

        # Create corpus and count word frequencies
        self.corpus = {}

        for i in range(number_of_items):
            # Tokenize the sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Iterate over every token
            for token in tk_sentence:
                if token in self.corpus:
                    # Check for sentiment
                    if self.sentiment[i] != 0:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1] + 1
                    else:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1]
                else:
                    # Check for sentiment
                    if self.sentiment[i] != 0:
                        self.corpus[token] = 1, 1
                    else:
                        self.corpus[token] = 1, 0

    def calcProbability(self):
        print 'Calculating unigram probability.'
        # Corpus created, calculate words probability of sentiment based on frequency
        self.probWord = {}
        for token in self.corpus.keys():
            self.probWord[token] = float(self.corpus[token][1]) / self.corpus[token][0]
            # print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'

        # Probability of sentiment per word calculated, estimate sentence probability of sentiment
        self.probSent = {}

        for i in range(len(self.sentence)):
                p = 1
                tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
                for token in tk_sent:
                    p = p + self.probWord[token]
                self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
                #print i, 'PROB', self.probSent[i], 'SENT', self.sentiment[i]

    def trainPerceptron(self, sizetrain = 6000):
        print 'Training perceptron.'
        spsv = self.probSent.values()
        ssv  = self.sentiment.values()

        
        trainingSetKeys = zip(spsv[0:sizetrain])
        trainingSetVals = [x != 0 for x in ssv][0:sizetrain]

        trainingSet = dict()
        for i in range(sizetrain):
            trainingSet[trainingSetKeys[i]] = trainingSetVals[i]
        
        self.p.train(trainingSet)
        print 'Found threshold: ', self.p.threshold / self.p.weights[0]
        
    def printResults(self):
        t = self.p.threshold / self.p.weights[0]
        confusion = {}
        confusion["tp"] = 0
        confusion["tn"] = 0
        confusion["fp"] = 0
        confusion["fn"] = 0
        for i in range(len(self.sentence)):
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
        print 'Results: '
        print confusion
        print 'accuracy = ', float(confusion["tp"] + confusion["tn"]) / (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
        print 'precision = ', float(confusion["tp"]) / (confusion["tp"] + confusion["fp"] )

m = Main()
