# import nltk
import csv
import nltk
import perceptron
import random
import time

class Main():

    # Open a file
    file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')     

    # Initialize dictionaries
    sentence = {}
    sentiment = {}
    corpus = {}
    probWord = {}
    probSent = {}
    wordVectors = {}
    
    # Perceptron used for machine learning
    p = perceptron.Perceptron()    

    # Initialize lists
    trainSet = []
    testSet = []
    validationSet = []
    bagOfWords = []

    # The distribution of all the data over train-, test- and validationset 
    distribution = (0.7, 0.1, 0.2)

    def __init__(self, iterations = 10):
        # Load the sentences and sentiments from file
        self.initializeCorpus()
        
        # Use machine learning method
#        self.singleInputPerceptron()
        self.multiInputPerceptron()

    def singleInputPerceptron(self):
        # Reset totals
        acc = 0
        pre = 0
        
        # Get current time
        t = time.time()
        
        for i in range( iterations ):
            print "--- iteration", i + 1, "of", iterations, "---"

            # Reset dictionaries
            self.corpus = {}
            self.probWord = {}
            self.probSent = {}
            self.p.reset()
            self.trainSet = []
            self.testSet = []
            self.validationSet = []
            
            # Go through the steps
            self.makeCorpus()
            self.calcProbability()
            self.trainSingleInputPerceptron()

            # Retrieve results
            result = self.printResults()

            # Add to the totals
            acc += result[0]
            pre += result[1]

        # Average results and print
        print 'Accuracy , averaged: ', acc / float(iterations)
        print 'Precision, averaged: ', pre / float(iterations)
        print 'Time taken for', iterations, 'iterations: ', time.time()- t

    def multiInputPerceptron(self):
        # Get current time
        t = time.time()
        
        # Go through the steps
        self.makeCorpus()
        self.createWordVectors()
        self.calcProbability()
        self.trainMultiInputPerceptron()
        self.testMultiInputPerceptron()

        print 'Time taken: ', time.time() - t

    def initializeCorpus(self):
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
            
            # Stop at 10000
            i += 1
            if ( i == 2000 ): break

    def makeCorpus(self):    
        print 'Creating corpus...'
        for i in range(1,2000):
            # Assign at random to train, test or validation set
                r = random.random()
                if ( r < self.distribution[0] ):
                    self.trainSet.append(i-1)
                else:
                    self.testSet.append(i-1)
            
        # Create corpus and count word frequencies
        self.corpus = {}
        
        for j in self.trainSet:
            # Tokenize the sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[j] )

            # Check for sentiment 
            sent = self.sentiment[j]

            # Iterate over every token
            for token in tk_sentence:
                # Add token to the bag of words
                if not token in self.bagOfWords:
                    self.bagOfWords.append(token)

                # Increment frequencies
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
#            print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'

    def calcProbability(self):
        # Probability of sentiment per word calculated, estimate sentence probability of sentiment
        self.probSent = {}

        for i in self.trainSet:
                p = 1
                tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
                for token in tk_sent:
                    p = p + self.probWord[token]
                self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
#                print i, 'PROB', self.probSent[i], 'SENT', self.sentiment[i]
        
    def trainSingleInputPerceptron(self):
        print 'Training perceptron.'
        ssv  = [x != 0 for x in self.sentiment.values()]
                
        trainingSet = {}
        for i in self.trainSet:
            trainingSet[(self.probSent[i],)] = ssv[i]
                 
        self.p.train(trainingSet)
        print 'Found threshold: ', self.p.threshold / self.p.weights[0]

        print 'Validating perceptron.'

        # Calculate probability for test sentences
        for i in self.testSet:
            p = 1
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            for token in tk_sent:
                try:
                    p = p + self.probWord[token]
                except:
                    # If word does not occur in corpus, ignore for now
                    # (can try katz backoff later?)
                    pass
            # Store the probability in  dictionary     
            self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
#            print i, 'PROB', self.probSent[i], 'SENT', self.sentiment[i]

    def trainMultiInputPerceptron(self):
        ssv  = [x != 0 for x in self.sentiment.values()]

        print "Number of inputs = ", len(self.bagOfWords)

        # Create trainingset for perceptron
        trainingSet = {}
        for i in self.trainSet:
            trainingSet[tuple(self.wordVectors[i])] = ssv[i]

        # Initialise weights on word probability
        print "Setting weights"
        self.p.set_weights([self.probWord[token] for token in self.bagOfWords])
        
        # Train the perceptron
        print 'Training perceptron'        
        self.p.train(trainingSet,0.1,100)
        
    def testMultiInputPerceptron(self):
        # Initialize confusion matrix
        confusion = {}
        confusion["tp"] = 0
        confusion["tn"] = 0
        confusion["fp"] = 0
        confusion["fn"] = 0

        print "Testing perceptron"
        
        for i in self.testSet:
            # Tokenize sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Create word vector
            vec = [ (x in tk_sentence) for x in self.bagOfWords]

            # Let the perceptron do its work
            out = self.p.output(vec)

            if out:
                if self.sentiment[i] == 0:
                    confusion["fp"] += 1
                else:
                    confusion["tp"] += 1
            else:
                if self.sentiment[i] == 0:
                    confusion["tn"] += 1
                else:
                    confusion["fn"] += 1

        # Print results                   
        print 'Results for test set: '
        print confusion
		
        try:
            acc = float(confusion["tp"] + confusion["tn"]) / (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
        except:
            acc = 0
        print 'accuracy = ', acc
        try:
            pre = float(confusion["tp"]) / (confusion["tp"] + confusion["fp"] )
        except:
            pre = 0
        print 'precision = ', pre

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
#        print 'Results for test set: '
#        print confusion
        acc = float(confusion["tp"] + confusion["tn"]) / (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
#        print 'accuracy = ', acc
        pre = float(confusion["tp"]) / (confusion["tp"] + confusion["fp"] )
#        print 'precision = ', pre
        return (acc, pre)        

    def createWordVectors(self):
        for i in self.trainSet:
            # Tokenize sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Create word vector
            vec = [ (x in tk_sentence) for x in self.bagOfWords]
            
#            for word in self.bagOfWords:
#                if word in tk_sentence:
#                    vec.append(1)
#                else:
#                    vec.append(0)
            self.wordVectors[i] = vec

m = Main(10)
