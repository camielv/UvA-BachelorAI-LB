# Extractor extracts a column from a csv file
import csv
import nltk
import perceptron
import random
import time
import re

class Main():

    # Open a file
    file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')

    # Initialize dictionaries
    sentence = {}
    sentiment = {}
    corpus = {}
    probWord = {}
    probSent = {}

    # Perceptron used for machine learning
    p = perceptron.Perceptron()

    # Initialize lists
    trainSet = []
    testSet = []
    validationSet = []

    # The distribution of all the data over train-, test- and validationset
    distribution = (0.7, 0.1, 0.2)

    # The number of sentences
    num_sentences = 0

    def __init__(self, iterations = 10):
        # Reset totals
        acc = 0
        pre = 0

        # Get current time
        t = time.time()

        #ngrams
        n = 3
        
        corpus = {}

        # Load the sentences and sentiments from file
        self.initializeCorpus( n )
        
        for i in range( iterations ):
            print "--- iteration", i + 1, "of", iterations, "---"
            
            # Reset dictionaries
            self.probWord = {}
            self.probSent = {}
            self.p.reset()
            self.trainSet = []
            self.testSet = []
            self.validationSet = []

            # Go through the steps
            self.makeCorpus( n )
            self.trainPerceptron( n )

            # Retrieve results
            result = self.printResults()

            # Add to the totals
            acc += result[0]
            pre += result[1]

        # Average results and print
        print 'Accuracy , averaged: ', acc / float(iterations)
        print 'Precision, averaged: ', pre / float(iterations)
        print 'Time taken for', iterations, 'iterations: ', time.time()- t
        
    def initializeCorpus(self, n, blogs=False):
        self.sentence = {}
        self.sentiment = {}

        # Initialize counter
        i = 0

        # Create corpus and count word frequencies
        self.corpus = {}
        print 'Creating corpus with ', n , '- grams.'

        # Collect sentences and sentiments
        for entry in self.file1:
            # Do not include header
            if i == 0:
                i+=1
                continue

            # Check for blogposts
            if not blogs:
                if int(entry[3]) != 3:
                    continue
            
            # The actual message is the 9th attribute, sentiment is the 4th
            curSent = re.sub('\||#|:|;|RT|@\w+|\**', '', entry[9])
            sent = float(entry[4])

            self.sentence[i - 1] = curSent
            self.sentiment[i - 1] = sent
            
            # Tokenize the sentence
            tk_sent = nltk.tokenize.word_tokenize( curSent )
       
            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                
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
               
            # Stop at 10000
            i += 1
            if ( i == 10000 ):
                break

        # Set the number of sentences
        self.num_sentences = i
        print 'Number of sentences =', self.num_sentences
        
    def makeCorpus(self, n):
        for i in range(1,self.num_sentences):
            # Assign at random to train, test or validation set
                r = random.random()
                if ( r < self.distribution[0] ):
                    self.trainSet.append(i-1)
                else:
                    self.testSet.append(i-1)
            
        print 'Calculating unigram probability'
        self.probWord = {}
        # Corpus created, calculate words probability of sentiment based on frequency
        for i in self.trainSet:
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            p = 0

            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                self.probWord[token] = float(self.corpus[token][1]) / self.corpus[token][0]
                # print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'
                p = p + self.probWord[token]
            self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
        
    def trainPerceptron(self, n):
        print 'Training perceptron'
        ssv = [x != 0 for x in self.sentiment.values()]
                
        trainingSet = {}
        for i in self.trainSet:
            trainingSet[(self.probSent[i],)] = ssv[i]
                 
        self.p.train(trainingSet)
        print 'Found threshold: ', self.p.threshold / self.p.weights[0]

        print 'Testing perceptron'

        # Calculate probability for test sentences
        for i in self.testSet:
            p = 0
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                try:
                    p = p + self.probWord[token]
                except:
                    # If word does not occur in corpus, ignore for now
                    # (can try katz backoff later?)
                    pass
            # Store the probability in dictionary
            self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
            # print i, 'PROB', self.probSent[i], 'SENT', self.sentiment[i]

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
# print 'Results for test set: '
# print confusion
        acc = float(confusion["tp"] + confusion["tn"]) / (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
# print 'accuracy = ', acc
        pre = float(confusion["tp"]) / (confusion["tp"] + confusion["fp"] )
# print 'precision = ', pre
        return (acc, pre)

m = Main(10)
