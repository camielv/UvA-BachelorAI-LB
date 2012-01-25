# import nltk
import csv
import nltk
import perceptron
import random
import time
import re
#from svmutil import *
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
    wordVectors = {}
    
    # Perceptrons used for machine learning
    p1 = perceptron.Perceptron()    
    p2 = perceptron.Perceptron()
    
    # Initialize lists
    trainSet = []
    testSet = []
    validationSet = []
    bagOfWords = []

    # The number of sentences
    num_sentences = 0

    def __init__(self):
        
        # Choose machine learning method
        self.singleInputPerceptron()
#        self.multiInputPerceptron()
#        self.supportVectorMachine()

    '''
        Machine learning methods:

    '''
    def singleInputPerceptron(self, iterations=1, total_messages = 10000):
        # Reset totals
        accNeu = 0
        preNeu = 0
        recNeu = 0
        
        accPos = 0
        prePos = 0
        recPos = 0
        
        # Get current time
        now = time.time()

        # The n for the n-grams
        n = 3
        
        # Load the sentences and sentiments from file
        self.initializeCorpus( n, total_messages )
        
        for i in range( iterations ):
            print "--- iteration", i + 1, "of", iterations, "---"
            
            # Reset variables
            self.probWord = {}
            self.probSent = {}
            self.trainSet = []
            self.testSet = []
            self.validationSet = []
            self.p1.reset()
            self.p2.reset()

            # Random selection of training and test data
            self.makeCorpus( n, distribution = (0.7, 0.3) )

            # Go through the steps to seperate opinion from nonopinion
            t = self.trainSingleInputPerceptron( n )
            # then seperate positive from negative in the test set of the previous step
            
            # Retrieve results
            result = self.printResults( t )

            # Add to the totals
            accNeu += result[0][0]
            preNeu += result[0][1]
            recNeu += result[0][2]
            
            accPos += result[1][0]
            prePos += result[1][1]
            recPos += result[1][2]

        # Average results and print
        print 'Neutral-vs-sentimented classifier:'
        print 'Accuracy , averaged: ', accNeu / float(iterations)
        print 'Precision, averaged: ', preNeu / float(iterations)
        print 'Recall,    averaged: ', recNeu / float(iterations)

        print 'Positive-vs-negative classifier:'
        print 'Accuracy , averaged: ', accPos / float(iterations)
        print 'Precision, averaged: ', prePos / float(iterations)
        print 'Recall,    averaged: ', recPos / float(iterations)
        
        print 'Time taken for', iterations, 'iterations: ', time.time()- now
            
    '''
        Corpus methods
    '''        
    def initializeCorpus(self, n, max_num = 10000,tweet_only=True):
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
    
        for i in self.trainSet:
            # Tokenize the sentence
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
       
            # Create temporary dictionary of dictionaries of lists
            temp_ngram = {}

            for k in range( 1, n + 1 ):
                temp_ngram[k] = {}
                for j in range( 1, k + 1 ):
                    temp_ngram[k][j] = []

            count = 0;
            # Iterate over every word
            for word in tk_sent:
                count += 1
                # Loop over every n-gram
                for k in range( 1, n + 1 ):
                    # Loop over every temporary instantion of an n gram
                    for j in range( 1, k + 1 ):
                        # Add this word
                        if count >= j:
                            temp_ngram[k][j].append(word)
                        
                        if len( temp_ngram[k][j] ) == k:
                            # We found a n-gram
                            token = tuple(temp_ngram[k][j])

                            # format: corpus[<combination of n tokens>]{neutrals, positives, negatives}
                            if token in self.corpus:
                                if self.sentiment[i] > 0:
                                    self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1] + 1, self.corpus[token][2]
                                elif self.sentiment[i] == 0:
                                    self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1], self.corpus[token][2]
                                else:
                                    self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1], self.corpus[token][2] + 1
                            else:
                                if self.sentiment[i] > 0:
                                    self.corpus[token] = 1, 1, 0
                                elif self.sentiment[i] == 0:
                                    self.corpus[token] = 1, 0, 0
                                else:
                                    self.corpus[token] = 1, 0, 1
                                    
                            temp_ngram[k][j] = []

            
        print 'Calculating unigram probability'



        self.probWord = {'Neutral':{}, 'Positive':{}}
        self.probSent = {'Neutral':{}, 'Positive':{}}
        # Corpus created, calculate words probability of sentiment based on frequency
        for i in self.trainSet:
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            # Create temporary dictionary of dictionaries of lists
            temp_ngram = {}

            for k in range( 1, n + 1 ):
                temp_ngram[k] = {}
                for j in range( 1, k + 1 ):
                    temp_ngram[k][j] = []

            count = 0;
            # Iterate over every word
            for word in tk_sent:
                count += 1
                # Loop over every n-gram
                for k in range( 1, n + 1 ):
                    # Loop over every temporary instantion of an n gram
                    for j in range( 1, k + 1 ):
                        # Add this word
                        if count >= j:
                            temp_ngram[k][j].append(word)
                        
                        if len( temp_ngram[k][j] ) == k:
                            # We found a n-gram
                            token = tuple(temp_ngram[k][j])
                            
                            # Chance of being neutral (not-opinion) is sum of positive and negative uses / total 
                            self.probWord['Neutral'][token] = float(self.corpus[token][1] + self.corpus[token][2]) / self.corpus[token][0]
                
                            # Chance of being positive (not-negative) is positives - negatives / total
                            self.probWord['Positive'][token]  = float(self.corpus[token][1] - self.corpus[token][2]) / self.corpus[token][0]
                            
                            temp_ngram[k][j] = []
                            
            pNeutral  = 0
            pPositive = 0
                         
            for j in range(len(tk_sent) - (n-1)):
                token = tuple(tk_sent[j:j+n])

                # increment chances according to occurrence                
                pNeutral  += self.probWord['Neutral'][token]                
                pPositive += self.probWord['Positive'][token]
                
            self.probSent['Neutral'][i] = pNeutral / float(len(tk_sent))
            self.probSent['Positive'][i] = pPositive / float(len(tk_sent))

    '''
        Train + test of methods:
    '''        
    def trainSingleInputPerceptron(self, n, print_scatter=False):
        print 'Training perceptron'
        # Create a list with 1 if opinion and 0 if non-opinion
        ssvNeu = [x != 0 for x in self.sentiment.values()]
        
        # Create a list with 1 if positive and 0 if negative
        ssvPos= [0 if x < 0 else 1 for x in self.sentiment.values()]
        
        # trainingset for opinion vs non-opinion classifier                
        trainingSet1 = {}
        trainingSet2 = {}
        j = 0
        for i in self.trainSet:
            trainingSet1[i] = ((self.probSent['Neutral'][i],), ssvNeu[i])
            # for the positive/negative threshold, only train on messages that do contain an opinion
            if self.sentiment[i] != 0:
                j += 1
                trainingSet2[j] = ((self.probSent['Positive'][i],), ssvPos[i])

        # train perceptron to find threshold         
        self.p1.train(trainingSet1)
        self.p2.train(trainingSet2)

        opinionthreshold = self.p1.threshold / self.p1.weights[0]
        positivethreshold = self.p2.threshold / self.p2.weights[0]

        print 'Found threshold for opinion-vs-nonopinion classifier: ', opinionthreshold
        print 'Found threshold for positive-vs-negative classifier: ', positivethreshold

        if print_scatter:
            temp_sentiment = []
            for i in self.trainSet:
                temp_sentiment.append( self.sentiment[i] )
            print len(temp_sentiment), ' =?= ', len(self.probSent['Neutral'])
            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self.probSent['Neutral'].values(),temp_sentiment)
            ax.axvline(x=opinionthreshold, color='r')
            ax.set_xlabel('Sentence Probability')
            ax.set_ylabel('Sentiment')
            plt.show()
            '''
        print 'Testing thresholds'

        # Calculate probability for test sentences
        for i in self.testSet:
            pNeutral = 0
            pPositive = 0
            
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])

                try:
                    pNeutral = pNeutral + self.probWord['Neutral'][token]
                    pPositive = pPositive + self.probWord['Positive'][token]
                except:
                    # If word does not occur in corpus, ignore for now
                    # (can try smaller n-grams later?)
                    pass

            # Store the probability in dictionary
            if pNeutral / float(len(tk_sent)) > 1:
                print self.sentence[i]
            self.probSent['Neutral'][i] = pNeutral / float(len(tk_sent)) 
            self.probSent['Positive'][i] = pPositive / float(len(tk_sent))
        return (opinionthreshold, positivethreshold)
    
    def printResults(self, thresholds):

        # dictionary containing number of true postives etc. for classifiers Positive and Neutral 
        confusion = {}
        confusion['Positive'] = {'tp':0,'tn':0,'fp':0,'fn':0}
        confusion['Neutral'] = {'tp':0,'tn':0,'fp':0,'fn':0}
        
        for i in self.testSet:
                if self.probSent['Neutral'][i] >= thresholds[0]:
                    if self.sentiment[i] == 0:
                        confusion['Neutral']['fp'] += 1
                        #print self.sentence[i], ' Distance = ', self.probSent[i], '-', self.sentiment[i], ' = ', self.probSent[i]- self.sentiment[i]
                    else:
                        confusion['Neutral']['tp'] += 1
                elif self.probSent['Neutral'][i] < thresholds[0]:
                    if self.sentiment[i] == 0:
                        confusion['Neutral']['tn'] += 1
                    else:
                        confusion['Neutral']['fn'] += 1
                
                # only test pos/neg for sentimental sentences
                if self.sentiment[i] != 0:
                    if self.probSent['Positive'][i] >= thresholds[1]:
                        if self.sentiment[i] < 0 :
                            confusion['Positive']['fp'] += 1
                        else:
                            confusion['Positive']['tp'] += 1
                    elif self.probSent['Positive'][i] < thresholds[1]:
                        if self.sentiment[i] < 0:
                            confusion['Positive']['tn'] += 1
                        else:
                            confusion['Positive']['fn'] += 1
                        
        accNeu = float(confusion['Neutral']['tp'] + confusion['Neutral']['tn']) / (confusion['Neutral']['tp'] + confusion['Neutral']['tn'] + confusion['Neutral']['fp'] + confusion['Neutral']['fn'])
        try:
            preNeu = float(confusion['Neutral']['tp']) / (confusion['Neutral']['tp'] + confusion['Neutral']['fp'] )
        except:
            preNeu = 0
        recNeu = float(confusion['Neutral']['tp']) / (confusion['Neutral']['tp'] + confusion['Neutral']['fn'] )

        accPos = float(confusion['Positive']['tp'] + confusion['Positive']['tn']) / (confusion['Positive']['tp'] + confusion['Positive']['tn'] + confusion['Positive']['fp'] + confusion['Positive']['fn'])
        try:
            prePos = float(confusion['Positive']['tp']) / (confusion['Positive']['tp'] + confusion['Positive']['fp'] )
        except:
            prePos = 0
        recPos = float(confusion['Positive']['tp']) / (confusion['Positive']['tp'] + confusion['Positive']['fn'] )

        print confusion

        return ((accNeu, preNeu, recNeu), (accPos, prePos, recPos))        

m = Main()
