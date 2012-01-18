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
    
    # Perceptron used for machine learning
    p = perceptron.Perceptron()    

    # Initialize lists
    trainSet = []
    testSet = []
    validationSet = []
    bagOfWords = []

    # The distribution of all the data over train-, test- and validationset 
    distribution = (0.7, 0.1, 0.2)

    # The number of sentences
    num_sentences = 0

    def __init__(self):
        
        # Choose machine learning method
        self.singleInputPerceptron()
#        self.multiInputPerceptron()
#        self.supportVectorMachine()

    def singleInputPerceptron(self, iterations=10):
        # Reset totals
        acc = 0
        pre = 0

        # Get current time
        t = time.time()

        # The n for the n-grams
        n = 3
        
        # Load the sentences and sentiments from file
        self.initializeCorpus( n, 10000 )
        
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
            self.trainSingleInputPerceptron( n )

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
            if ( i == max_num ):
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
        
    def trainSingleInputPerceptron(self, n):
        print 'Training perceptron'
        ssv = [x != 0 for x in self.sentiment.values()]
                
        trainingSet = {}
        for i in self.trainSet:
            trainingSet[i] = ((self.probSent[i],), ssv[i])
                 
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

    def printResults(self):
        wrongness = dict()
        
        t = self.p.threshold / self.p.weights[0]
        confusion = {}
        confusion['tp'] = 0
        confusion['tn'] = 0
        confusion['fp'] = 0
        confusion['fn'] = 0
        for i in self.testSet:
                if self.probSent[i] > t:
                    if self.sentiment[i] == 0:
                        confusion['fp'] += 1
                        wrongness[i] = (self.sentence[i], self.probSent[i], self.sentiment[i], self.probSent[i]- self.sentiment[i])
                    else:
                        confusion['tp'] += 1
                if self.probSent[i] < t:
                    if self.sentiment[i] == 0:
                        confusion['tn'] += 1
                    else:
                        confusion['fn'] += 1
                        wrongness[i] = (self.sentence[i], self.probSent[i], self.sentiment[i], self.probSent[i]- self.sentiment[i])
#        print 'Results for test set: '
#        print confusion
        acc = float(confusion['tp'] + confusion['tn']) / (confusion['tp'] + confusion['tn'] + confusion['fp'] + confusion['fn'])
#        print 'accuracy = ', acc
        pre = float(confusion['tp']) / (confusion['tp'] + confusion['fp'] )
#        print 'precision = ', pre

        inp = open('wrongness.txt', 'w')
        for q in wrongness:
            inputding = 'Dist = ' + str(wrongness[q][1]) + ' - ' + str(wrongness[q][2]) + ' = ' + str(wrongness[q][3]) + '  ' + str(wrongness[q][0] + '\n') 
            inp.write(inputding)
        inp.close()
        return (acc, pre)        

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
