# import nltk
import csv
import nltk
import perceptron
import random
import time
import re
#from svmutil import *

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
    def singleInputPerceptron(self, iterations=10, total_messages = 10000):
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
            

    def multiInputPerceptron(self):
        # Load the sentences and sentiments from file
        self.initializeCorpus( 1 )
        
        # Get current time
        t = time.time()
        
        # Go through the steps
        self.makeCorpus( 1 )
        self.createWordVectors()
        self.calcProbability()
        self.trainMultiInputPerceptron()
        self.testMultiInputPerceptron()

        print 'Time taken: ', time.time() - t

    def supportVectorMachine(self,c=10):
        # Get current time
        t = time.time()
        
        self.initializeCorpus( 1, 400 )
        self.makeCorpus( 1 )
        self.createWordVectors()
      
        # Create file with data
        f = open('./SVM_data.txt', 'w')

        # Create the classes vector
        n = 0
        for i in self.trainSet:
            if self.sentiment[i] != 0:
                f.write('+1')
            else:
                f.write('-1')
            
            k = 0
            for j in self.wordVectors[i]:
                f.write(' {0}:{1}'.format(k,int(j)))
                k += 1
            f.write('\n')
            n += 1
        f.close()
        print n, 'lines written'
        
        # Train the model
        print 'Creating SVM problem'
        y, x = svm_read_problem('./SVM_data.txt')
        print 'Training SVM' 
        m = svm_train(y,x, '-c 10')

        

        # Testing the model
        print 'Testing the SVM'
        f = open('./SVM_test.txt', 'w')

        real_answer = []
        
        # Create the classes vector
        for i in self.testSet:
            
            if self.sentiment[i] != 0:
                f.write('+1')
                real_answer.append(1)
            else:
                f.write('-1')
                real_answer.append(-1)
            
            k = 0
            for j in self.wordVectors[i]:
                f.write(' {0}:{1}'.format(k,int(j)))
                k += 1
            f.write('\n')
        f.close()

        y1,x1 = svm_read_problem('./SVM_test.txt')
        p_label, p_acc, p_val = svm_predict(y1, x1, m)
        print 'Accuracy and mean squared error: {1}'.format(p_label,p_acc)
        print 'Time taken: ', time.time() - t

        print 'Validating results'
        confusion = {}
        confusion['tp'] = 0
        confusion['tn'] = 0
        confusion['fp'] = 0
        confusion['fn'] = 0
        
        for x,y in zip(p_label,real_answer):
            if x == 1:
                if y == -1:
                    confusion['fp'] += 1
                else:
                    confusion['tp'] += 1
            else:
                if y == -1:
                    confusion['tn'] += 1
                else:
                    confusion['fn'] += 1

        # Print results                   
        print 'Results for test set: '
        print confusion
        try:
            acc = float(confusion['tp'] + confusion['tn']) / (confusion['tp'] + confusion['tn'] + confusion['fp'] + confusion['fn'])
        except:
            acc = 0
        print 'accuracy = ', acc
        try:
            pre = float(confusion['tp']) / (confusion['tp'] + confusion['fp'] )
        except:
            pre = 0
        print 'precision = ', pre

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
            
            # Tokenize the sentence
            tk_sent = nltk.tokenize.word_tokenize( curSent )
       
            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])

                # format: corpus[<combination of n tokens>]{neutrals, positives, negatives}
                if token in self.corpus:
                    if sent > 0:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1] + 1, self.corpus[token][2]
                    elif sent == 0:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1], self.corpus[token][2]
                    else:
                        self.corpus[token] = self.corpus[token][0] + 1, self.corpus[token][1], self.corpus[token][2] + 1
                else:
                    if sent > 0:
                        self.corpus[token] = 1, 1, 0
                    elif sent == 0:
                        self.corpus[token] = 1, 0, 0
                    else:
                        self.corpus[token] = 1, 0, 1
               
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
            
        print 'Calculating unigram probability'

        self.probWord = {'Neutral':{}, 'Positive':{}}
        self.probSent = {'Neutral':{}, 'Positive':{}}
        # Corpus created, calculate words probability of sentiment based on frequency
        for i in self.trainSet:
            tk_sent = nltk.tokenize.word_tokenize( self.sentence[i] )
            pNeutral  = 0
            pPositive = 0
            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])

                # Chance of being neutral (not-opinion) is sum of positive and negative uses / total 
                self.probWord['Neutral'][token] = float(self.corpus[token][1] + self.corpus[token][2]) / self.corpus[token][0]
                # increment chances according to occurrence                
                pNeutral  += self.probWord['Neutral'][token]

                # Chance of being positive (not-negative) is positives - negatives / total
                self.probWord['Positive'][token]  = float(self.corpus[token][1] - self.corpus[token][2]) / self.corpus[token][0]
                pPositive += self.probWord['Positive'][token]

            # Calculate sentence probability for both classes, float division necessary here
            self.probSent['Neutral'][i] = pNeutral / float(len(tk_sent)) 
            self.probSent['Positive'][i] = pPositive / float(len(tk_sent))

    '''
        Train + test of methods:
    '''        
    def trainSingleInputPerceptron(self, n):
        print 'Training perceptron'
        # Create a list with 1 if opinion and 0 if non-opinion
        ssvNeu = [x != 0 for x in self.sentiment.values()]
        
        # Create a list with 1 if positive and 0 if negative
        ssvPos= [0 if x < 0 else 1 for x in self.sentiment.values()]
        print self.sentiment.values()[0:10]
        print ssvPos[0:10]
        
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
            self.probSent['Neutral'][i] = pNeutral / float(len(tk_sent)) 
            self.probSent['Positive'][i] = pPositive / float(len(tk_sent))
        return (opinionthreshold, positivethreshold)
    
    def trainMultiInputPerceptron(self):
        ssv  = [x != 0 for x in self.sentiment.values()]

        print 'Number of inputs = ', len(self.bagOfWords)

        # Create trainingset for perceptron
        trainingSet = {}
        for i in self.trainSet:
            trainingSet[i] = (tuple(self.wordVectors[i]), ssv[i])

        # Initialise weights on word probability
        print 'Setting weights'
        self.p.set_weights([self.probWord[token] for token in self.bagOfWords])
        
        # Train the perceptron
        print 'Training perceptron'        
        self.p.train(trainingSet,0.1,100)
        
    def testMultiInputPerceptron(self):
        # Initialize confusion matrix
        confusion = {}
        confusion['tp'] = 0
        confusion['tn'] = 0
        confusion['fp'] = 0
        confusion['fn'] = 0

        print 'Testing perceptron'
        
        for i in self.testSet:
            # Tokenize sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Create word vector
            vec = [ (x in tk_sentence) for x in self.bagOfWords]

            # Let the perceptron do its work
            out = self.p.output(vec)

            if out:
                if self.sentiment[i] == 0:
                    confusion['fp'] += 1
                else:
                    confusion['tp'] += 1
            else:
                if self.sentiment[i] == 0:
                    confusion['tn'] += 1
                else:
                    confusion['fn'] += 1

        # Print results                   
        print 'Results for test set: '
        print confusion
        try:
            acc = float(confusion['tp'] + confusion['tn']) / (confusion['tp'] + confusion['tn'] + confusion['fp'] + confusion['fn'])
        except:
            acc = 0
        print 'accuracy = ', acc
        try:
            pre = float(confusion['tp']) / (confusion['tp'] + confusion['fp'] )
        except:
            pre = 0
        print 'precision = ', pre

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
        preNeu = float(confusion['Neutral']['tp']) / (confusion['Neutral']['tp'] + confusion['Neutral']['fp'] )
        recNeu = float(confusion['Neutral']['tp']) / (confusion['Neutral']['tp'] + confusion['Neutral']['fn'] )

        accPos = float(confusion['Positive']['tp'] + confusion['Positive']['tn']) / (confusion['Positive']['tp'] + confusion['Positive']['tn'] + confusion['Positive']['fp'] + confusion['Positive']['fn'])
        prePos = float(confusion['Positive']['tp']) / (confusion['Positive']['tp'] + confusion['Positive']['fp'] )
        recPos = float(confusion['Positive']['tp']) / (confusion['Positive']['tp'] + confusion['Positive']['fn'] )

        print confusion

        return ((accNeu, preNeu, recNeu), (accPos, prePos, recPos))        

    def createWordVectors(self):
        # Create the bag of words
        print 'Creating the bag of words'
        for i in self.trainSet:
            # Tokenize sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Check all tokens
            for token in tk_sentence:
                if token not in self.bagOfWords:
                    self.bagOfWords.append(token)
            
        # Create the word vectors
        print 'Creating word vectors'
        for i in self.trainSet:
            # Tokenize sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Create word vector
            vec = [ (x in tk_sentence) for x in self.bagOfWords]
            
            self.wordVectors[i] = vec
            
        for i in self.testSet:
            # Tokenize sentence
            tk_sentence = nltk.tokenize.word_tokenize( self.sentence[i] )

            # Create word vector
            vec = [ (x in tk_sentence) for x in self.bagOfWords]
            
            self.wordVectors[i] = vec

m = Main()
