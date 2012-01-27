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
    probSent = {}
    corpus = {}
    probWord = {}
    naiveBayes = {}
    
    # Initialize lists
    trainSet = []
    testSet = []

    # The number of sentences
    num_sentences = 0

    def __init__(self):
        self.maxEntClassifier(1)

    '''
        Machine learning methods:

    '''
    def maxEntClassifier(self, iterations=10, total_messages = 10000, print_scatter=False):        
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
            self.naiveBayes = {}
            self.sentenceLabel = {}
            self.trainSet = []
            self.testSet = []
            self.priorNeutral  = 0
            self.priorPositive = 0
            self.priorNegative = 0
            self.featureSet= []

            # Random selection of training and test data
            self.makeCorpus( n, distribution = (0.7, 0.3) )
            print 'Start training'
            self.MEC = nltk.classify.MaxentClassifier.train( self.featureSet, 'IIS', max_iter=10 )
            
            print 'Start testing'
            self.testNaiveBayes()
            
            # Testing
            temp_set = [],[]
            for j in self.testSet:
                tk_sent = self.sentence[j]
            
                # If sentence is longer than 3
                if len(tk_sent) >= 3:
                    temp_set[1].append( self.sentiment[j] )
                    temp_set[0].append( self.sentenceLabel[j] )
            # Create confusion matrix
            confusion = [[0,0,0],[0,0,0],[0,0,0]]
            for j in range(len(temp_set[0])):
                sent = temp_set[1][j]
                clas = temp_set[0][j]

                if sent < 0: sent = -1
                if sent > 0: sent = 1

                confusion[int(sent+1)][int(clas+1)] += 1
                
            print confusion
            for j in range(3):
                for k in range(3):
                    allconfusion[j][k] += confusion[j][k]

        # Calculate mean
        print 'Average:'
        total = 0
        for i in range(3):
            for j in range(3):
                allconfusion[i][j] /= float(iterations)
                total += allconfusion[i][j]

        #acc = (allconfusion[0][0] + allconfusion[1][1] + allconfusion[2][2]) / float(total)
        # for every real class
        s = 0
        for i in range(3):
            row_sum = sum(allconfusion[i])
            s+= row_sum
            
        for i in range(3):
            row_sum = sum(allconfusion[i])
            col_sum = 0
            for j in range(3):
                col_sum += confusion[j][i]
                
            truepositives = float(allconfusion[i][i])
            truenegatives = s - row_sum - col_sum + truepositives
            falsepositives = col_sum - truepositives
            falsenegatives = row_sum - truepositives

            print 'For {0}-class:'.format(['Negative','Neutral','Positive'][i])
            
            rec = truepositives / (truepositives + falsenegatives )
            pre = truepositives / ( truepositives + falsepositives )
            acc = (truepositives + truenegatives) / s
 
            print 'Recall', rec
            print 'Precision', pre
            print 'Accuracy', acc
        print allconfusion

        print 'Time taken for', iterations, 'iterations: ', time.time()- now

    '''
        Cleaning methods
    '''
    def clean( self, sentence ):
        # print sentence
        sentence = sentence.replace( ':-)', " blijesmiley " )
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( ':s', ' awkwardsmiley ' )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )
        
        # delete non-expressive words
        #sentence = re.sub('en|de|het|ik|jij|zij|wij|deze|dit|die|dat|is|je|na|zijn|uit|tot|te|sl|hierin|naar|onder', '', sentence)
        
        # Delete expressions, such as links, hashtags, twitteraccountnames 
        sentence = re.sub( r'\:P|\:p|http\/\/t\.co\/\w+|\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )
        
        # remove double letters
        #for x in 'abcdefghijklmnopqrstuvwxyz':
        #    sentence = re.sub(x+'+', x, sentence )

        return sentence
        # print sentence
        # Werkt nog niet cleanup is nog niet goed genoeg
        return self.__stemmer.stem( sentence )

    def tokenize( self, sentence ):  
        return re.findall('\w+|\?|\!', sentence)
            
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
            curSent = self.tokenize( self.clean( entry[9] ) )
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

        self.featureSet = []
        # Count for every class the occurences
        for i in self.trainSet:
            tk_sent = self.sentence[i]

            # If sentence is longer than n
            if len(tk_sent) >= n:
                features = {}
                if self.sentiment[i] < 0:
                    label = -1
                elif self.sentiment[i] > 0:
                    label = 1
                else:
                    label = 0
            
                
                # Iterate over every n tokens
                for j in range( len(tk_sent) - (n-1) ):
                    # token is now a uni/bi/tri/n-gram instead of a token
                    token = tuple( tk_sent[j:j+n] )
                    if token in features:
                        features[token] += 1
                    else:
                        features[token] = 1
                
                self.featureSet.append( (features,label) )    
                    
                
    def testNaiveBayes(self, n=3):
         for i in self.testSet:
            tk_sent = self.sentence[i]

            # If sentence is longer than n
            if len(tk_sent) >= n:
                features = {}
                label = self.sentiment[i]
            
                
                # Iterate over every n tokens
                for j in range( len(tk_sent) - (n-1) ):
                    # token is now a uni/bi/tri/n-gram instead of a token
                    token = tuple( tk_sent[j:j+n] )
                    if token in features:
                        features[token] += 1
                    else:
                        features[token] = 1

                self.sentenceLabel[i] = self.MEC.classify(features)
m = Main()
