import csv
import time
import re
import random
import math
from svmutil import *

# Open a file
file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')


def clean( sentence ):
    sentence = sentence.replace( ':-)', " blijesmiley " )
    sentence = sentence.replace( ':)', " blijesmiley " )
    sentence = sentence.replace( ':(', " zieligesmiley " )
    sentence = sentence.replace( ':s', ' awkwardsmiley ' )
    sentence = sentence.replace( '!', " ! " )
    sentence = sentence.replace( '?', " ? " )
    
    # delete non-expressive words
    sentence = re.sub(' en | de | het | ik | jij | zij | wij | deze | dit | die | dat | is | je | na | zijn | uit | tot | te | sl | hierin | naar | onder ', '', sentence)
    
    # Delete useless info, such as links, hashtags, twitteraccountnames 
    sentence = re.sub( r'http\/\/t\.co\/\w+|\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
    sentence = re.sub( ' +',' ', sentence )
    return sentence
    # print sentence
    # Werkt nog niet cleanup is nog niet goed genoeg
    #return __stemmer.stem( sentence )

def tokenize( sentence ):
    return re.findall('\w+|\?|\!', sentence)

def initializeCorpus(n, max_num = 10000,tweet_only=True):
    sentence = {}
    sentiment = {}

    # Initialize counter
    i = 0

    # Create corpus and count word frequencies
    corpus = {}
    print 'Creating corpus with ', n , '- grams.'

    # Collect sentences and sentiments
    for entry in file1:
        # Do not include header
        if i == 0:
            i+=1	
            continue

        # Check for tweets
        if tweet_only:
            if int(entry[3]) != 3:
                continue
        
        # The actual message is the 9th attribute, sentiment is the 4th
        curSent = tokenize( clean(entry[9]) )
        sent = float(entry[4])

        sentence[i - 1] = curSent
        sentiment[i - 1] = sent
           
        # Stop at 10000
        i += 1
        if ( i == max_num ):
            i -= 1
            break
    # Set the number of sentences
    num_sentences = i
    print 'Number of sentences =', num_sentences

    return sentence, sentiment, num_sentences
    
def makeCorpus(n, num_sentences):
    
    testSet = list()
    trainSet = list()
    
    for i in range(1,num_sentences):
        # Assign at random to train, test or validation set
        r = random.random()
        if ( r < 0.7 ):
            trainSet.append(i-1)
        else:
            testSet.append(i-1)
    return testSet, trainSet

def neuralNetwork(iterations = 20):
    # Get current time
    now = time.time()
    
    senten = initializeCorpus( 1, 50 )
    (sentence, sentiment, num_sentences) = senten

    # create a corpus
    sets = makeCorpus( 1, num_sentences ) 
    trainSet = sets[1]
    testSet = sets[0]
    
    inputVector = createWordVectors(trainSet, num_sentences, sentence)

    # Network initialization
    nodes = dict()
    inputNodes = dict()
    layerNodes = dict()
    outputNodes = dict()

    num_hidden = len(inputVector)
    num_classes = 3
    
    # inputnodes have a value, weight
    # layernodes have a value, weight,
    # outputnodes only a value
    inputNodes = {'v':dict(), 'w':dict()}
    layerNodes = {'v':dict(), 'w':dict()}
    outputNodes = {'v':dict() }
    
    for iteration in range(iterations):
        print 'Iteration', iteration+1, ':'
        Delta = {0:dict(), 1:dict()}

        for t in trainSet:
            s = sentence[t]

            # initialize Delta
            for i in range(len(inputVector)):
                Delta[0][i] = dict()
                for j in range( num_hidden ):
                    Delta[0][i][j] = 0
                    Delta[1][j] = dict()
                    for k in range( num_classes ):
                        Delta[1][j][k] = 0

            # initialize input
            for i in range(len(inputVector)):
                # copy input values
                inputNodes['v'][i] = inputVector[t][i]

                inputNodes['w'][i] = dict()
                for j in range( num_hidden ):
                    # weight from node i to node j
                    inputNodes['w'][i][j] = 1
                    layerNodes['w'][j] = dict()
                    for k in range( num_classes ):
                        # weight from node j to k
                        layerNodes['w'][j][k] = 1

            # forward progapagation
            for j in range( num_hidden ):
                inputValue = 0
                for i in range( len( inputVector ) ):
                    inputValue += inputNodes['v'][j] * inputNodes['w'][i][j]
                layerNodes['v'][j] = (1  / (1 + math.exp( -inputValue )))  

            for k in range( num_classes ):
                inputValue = 0
                for j in range( num_hidden ):
                    inputValue += layerNodes['v'][k] * layerNodes['w'][j][k]
                outputNodes['v'][k] = (1  / (1 + math.exp( -inputValue )))

            # calculate error of each node delta: backward 
            delta = {1:dict(), 2:dict(), 3:dict()}
            # neutral
            delta[2][0] = outputNodes['v'][0] - (sentiment[t] == 0)
            # positive
            delta[2][1] = outputNodes['v'][1] - (sentiment[t] > 0)
            # negative
            delta[2][2] = outputNodes['v'][2] - (sentiment[t] < 0)

            for j in range( num_hidden ):
                delta[1][j] = 0
                for k in range( num_classes ):
                    delta[1][j] += layerNodes['w'][j][k] * delta[2][k] * ( layerNodes['v'][j] * ( 1 - layerNodes['v'][j] ))

            for i in range( len(inputVector )):
                for j in range( num_hidden ):
                    Delta[0][i][j] += inputNodes['v'][i] * delta[1][j] 
                    for k in range( num_classes ):
                        Delta[1][j][k] += layerNodes['v'][j] * delta[2][k]

        # update weights:
        alpha = 0.01 # learning rate
        
        for i in range( len( inputVector )):
            for j in range( num_hidden ):
                for k in range( num_classes ):
                    layerNodes['w'][j][k] += alpha * layerNodes['v'][j] * Delta[1][j][k]

            inputNodes['w'][i][j] += alpha * inputNodes['v'][i] * Delta[0][i][j]
    
        
    print 'Time taken: ', time.time() - now
    # test stuff
    for t in testSet:
        s = sentence[t]

        # initialize input
        for i in range(len(inputVector)):
            # copy input values
            inputNodes['v'][i] = inputVector[t][i]

            inputNodes['w'][i] = dict()
            for j in range( num_hidden ):
                # weight from node i to node j
                inputNodes['w'][i][j] = 1
                layerNodes['w'][j] = dict()
                for k in range( num_classes ):
                    # weight from node j to k
                    layerNodes['w'][j][k] = 1

        # forward progapagation
        for j in range( num_hidden ):
            inputValue = 0
            for i in range( len( inputVector ) ):
                inputValue += inputNodes['v'][j] * inputNodes['w'][i][j]
            layerNodes['v'][j] = (1  / (1 + math.exp( -inputValue )))  

        for k in range( num_classes ):
            inputValue = 0
            for j in range( num_hidden ):
                inputValue += layerNodes['v'][k] * layerNodes['w'][j][k]
            outputNodes['v'][k] = (1  / (1 + math.exp( -inputValue )))

        print 'True sentiment:', sentiment[t]
        print 'Found neutral/pos/neg values:', outputNodes['v'].values()
        

def createWordVectors(trainSet, num_sentences, sentence ):
    # Create the bag of words
    print 'Creating the bag of words'
    bagOfWords = list()
    for i in trainSet:
        # Tokenize sentence
        tk_sentence = sentence[i]

        # Check all tokens
        for token in tk_sentence:
            if token not in bagOfWords:
                bagOfWords.append(token)

    bagOfWords = set(bagOfWords)

    # Create the word vectors
    wordVectors = dict()
    print 'Creating word vectors'
    for i in range(len(sentence)):
        # Tokenize sentence
        tk_sentence = sentence[i]
        
        # Create word vector 
        vec = [ tk_sentence.count(x) for x in bagOfWords]
        
        wordVectors[i] = vec
    return wordVectors

neuralNetwork()
