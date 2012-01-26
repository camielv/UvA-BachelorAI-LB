import csv
import nltk
import random
import time
import re
import numpy as np
#import matplotlib.pyplot as plt

class Main():

    # Open a file
    file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')     

    # Initialize dictionaries
    sentence = {}
    sentiment = {}
    corpus = {}
    probWord = {}
    probSent = {}
    naiveBayes = {}
    
    # Initialize lists
    trainSet = []
    testSet = []

    # The number of sentences
    num_sentences = 0

    def __init__(self):
        self.naiveBayesClassifier(10)

    '''
        Machine learning methods:

    '''
    def naiveBayesClassifier(self, iterations=10, total_messages = 10000, print_scatter=False):        
        # Get current time
        now = time.time()

        # The n for the n-grams
        n = 3
        
        # Load the sentences and sentiments from file
        self.initializeCorpus( n, total_messages )


        allconfusion = [[0,0,0],[0,0,0],[0,0,0]]
        for i in range( iterations ):
            print "--- iteration", i + 1, "of", iterations, "---"
            
            # Reset variables
            self.probWord = {}
            self.probSent = {}
            self.trainSet = []
            self.testSet = []
            self.priorNeutral  = 0
            self.priorPositive = 0
            self.priorNegative = 0


            # Random selection of training and test data
            self.makeCorpus( n, distribution = (0.7, 0.3) )
            self.testNaiveBayes()
            
            # Testing
            temp_set = [],[]
            for j in self.testSet:
                tk_sent = nltk.tokenize.word_tokenize( self.sentence[j] )
            
                # If sentence is longer than 3
                if len(tk_sent) >= 3:
                    temp_set[1].append( self.sentiment[j] )
                    m = max(self.probSent[j])
                    if m == self.probSent[j][0]:
                        temp_set[0].append( 0 )
                    elif m == self.probSent[j][1]:
                        temp_set[0].append( 1 )
                    else:
                        temp_set[0].append( -1 )

            # Create confusion matrix
            confusion = [[0,0,0],[0,0,0],[0,0,0]]
            for j in range(len(temp_set[0])):
                sent = temp_set[1][j]
                clas = temp_set[0][j]

                if sent < 0: sent = -1
                if sent > 0: sent = 1

                confusion[int(sent+1)][int(clas+1)] += 1
                
            print confusion
            for n in range(3):
                for m in range(3):
                    allconfusion[n][m] += confusion[n][m]

        # Calculate mean
        print 'Average:'
        total = 0
        for n in range(3):
            for m in range(3):
                allconfusion[n][m] /= iterations
                total += allconfusion[n][m]

        #acc = (allconfusion[0][0] + allconfusion[1][1] + allconfusion[2][2]) / float(total)
        # for every real class
        s = 0
        for n in range(3):
            row_sum = sum(allconfusion[n])
            s+= row_sum
            
        for n in range(3):
            row_sum = sum(allconfusion[n])
            col_sum = 0
            for m in range(3):
                col_sum += confusion[m][n]
                
            truepositives = float(allconfusion[n][n])
            truenegatives = s - row_sum - col_sum + truepositives
            falsepositives = col_sum - truepositives
            falsenegatives = row_sum - truepositives

            print 'For class ', n , ':'
            
            rec = truepositives / (truepositives + falsenegatives )
            pre = truepositives / ( truepositives + falsepositives )
            acc = (truepositives + truenegatives) / s
 
            print 'Recall', rec
            print 'Precision', pre
            print 'Accuracy', acc
        print allconfusion

        print 'Time taken for', iterations, 'iterations: ', time.time()- now
            
    '''
        Corpus methods
    '''        
    def initializeCorpus(self, n, max_num=10000,tweet_only=True):
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

            # Check for tweets
            if tweet_only:
                if int(entry[3]) != 3:
                    continue
            
            # The actual message is the 9th attribute, sentiment is the 4th
            curSent = re.sub('\||#|:|;|RT|@\w+|\**', '', entry[9])
            sent = float(entry[4])

            self.sentence[i - 1] = curSent
            self.sentiment[i - 1] = sent
            # Stop at 10000
            i += 1
            if ( i == max_num ):
                break
        # Set the number of sentences
        self.num_sentences = i
        print 'Number of sentences =', self.num_sentences
            
    def makeCorpus( self, n, distribution ):
        for i in range(1,self.num_sentences):
            # Assign at random to train, test or validation set
            r = random.random()
            if ( r < distribution[0] ):
                self.trainSet.append(i-1)
            else:
                self.testSet.append(i-1)
        
        # Count for every class the occurences
        for i in self.trainSet:
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            
            # If sentence is longer than 3
            if len(tk_sent) >= 3:
                # Iterate over every n tokens
                for j in range( len(tk_sent) - (n-1) ):
                    # token is now a uni/bi/tri/n-gram instead of a token
                    token = tuple( tk_sent[j:j+n] )
                    if token in self.naiveBayes:
                        if self.sentiment[i] < 0:
                            self.naiveBayes[token][2] += 1
                        elif self.sentiment[i] > 0:
                            self.naiveBayes[token][1] += 1
                        else:
                            self.naiveBayes[token][0] += 1
                    else:
                        if self.sentiment[i] < 0:
                            self.naiveBayes[token] = [0,0,1]
                        elif self.sentiment[i] > 0:
                            self.naiveBayes[token] = [0,1,0]
                        else:
                            self.naiveBayes[token] = [1,0,0]
                
        print 'Calculating classes probability'

        sums = [0,0,0]
        for token in self.naiveBayes.keys():
            sums[0] += self.naiveBayes[token][0]
            sums[1] += self.naiveBayes[token][1]
            sums[2] += self.naiveBayes[token][2]
        totalSum = sum(sums)
        self.priorNeutral    = float(sums[0]) / totalSum
        self.priorPositive = float(sums[1]) / totalSum #(sums[1] + sums[2])
        self.priorNegative = float(sums[2]) / totalSum #(sums[1] + sums[2])

        print 'pNeu:{0}, pNNe:{1}, pNeg:{2}'.format(self.priorNeutral,self.priorPositive,self.priorNegative)

        self.probWord = {}
        self.probSent = {}
        
        # Corpus created, calculate words probability of sentiment based on frequency
        for i in self.trainSet:
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            
            # Iterate over every n tokens
            if len(tk_sent) >= 3:
                pNeutral  = self.priorNeutral
                pNegative = self.priorNegative
                pPositive = self.priorPositive

                for j in range( len(tk_sent) - (n-1) ):
                    token = tuple(tk_sent[j:j+n])
                    pNeutral  *= float(self.naiveBayes[token][0]) / sum(self.naiveBayes[token])
                    pPositive *= float(self.naiveBayes[token][1]) / sum(self.naiveBayes[token])
                    pNegative *= float(self.naiveBayes[token][2]) / sum(self.naiveBayes[token])
                        
                self.probSent[i] = (pNeutral,pPositive,pNegative)

    def testNaiveBayes(self, n=3):
        for i in self.testSet:
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            
            # If sentence is longer than 3
            if len(tk_sent) >= 3:
                pNeutral  = self.priorNeutral
                pNegative = self.priorNegative
                pPositive = self.priorPositive
                # Iterate over every n tokens

                for j in range( len(tk_sent) - (n-1) ):
                    token = tuple(tk_sent[j:j+n])
                    if token in self.naiveBayes:
                        pNeutral  *= float(self.naiveBayes[token][0]) / sum(self.naiveBayes[token])
                        pPositive *= float(self.naiveBayes[token][1]) / sum(self.naiveBayes[token])
                        pNegative *= float(self.naiveBayes[token][2]) / sum(self.naiveBayes[token])
                        
                self.probSent[i] = (pNeutral,pPositive,pNegative)
                        

m = Main()
