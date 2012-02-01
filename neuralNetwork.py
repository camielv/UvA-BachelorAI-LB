import csv
import time
import re
import random
import math
import sys
import cPickle as pickle
from svmutil import *

# Open a file
file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')
weights = dict()
    
def clean( sentence ):
    sentence = sentence.replace( ':-)', " blijesmiley " )
    sentence = sentence.replace( ':)', " blijesmiley " )
    sentence = sentence.replace( ':(', " zieligesmiley " )
    sentence = sentence.replace( ':s', ' awkwardsmiley ' )
    sentence = sentence.replace( '!', " ! " )
    sentence = sentence.replace( '?', " ? " )
    
    # Delete useless info, such as links, hashtags, twitteraccountnames 
    sentence = re.sub('RT|@\w+|http.*', '', sentence)
    sentence = re.sub( r'\.|\,|\/|\\', '', sentence )
    sentence = re.sub( r'\[|\]|&#39;s|\||#|:|;|\(|\)|\**', '', sentence )
    sentence = re.sub( ' +',' ', sentence )

    # delete non-expressive words
    sentence = re.sub(' he | op | wie | uit | EO | eo | en | de | het | ik | jij | zij | wij | deze | dit | die | dat | is | je | na | zijn | uit | tot | te | sl | hierin | naar | onder | is ', ' ', sentence)

    return sentence
    # print sentence
    # Werkt nog niet cleanup is nog niet goed genoeg
    #return __stemmer.stem( sentence )

def tokenize( sentence ):
    return re.findall('\w+', sentence)

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

def neuralNetwork( iterations = 1000, filename = './weights.txt'):
       
    
    # Get current time
    now = time.time()
    
    senten = initializeCorpus( 1, 100 )

    (sentence, sentiment, num_sentences) = senten

    # create a corpus
    sets = makeCorpus( 1, num_sentences ) 
    trainSet = sets[1]
    testSet = sets[0]
    #trainSet = [0]
    #testSet = [0]
        
    inputVector, bagOfWords = createWordVectors( trainSet, num_sentences, sentence )    

    # Network initialization
    nodes = dict()
    inputNodes = dict()
    layerNodes = dict()
    outputNodes = dict()


    num_hidden = 5
    num_classes = 1
    
    '''
    trainSet = [0,1,2,3]
    testSet = [0,1,2,3]
    inputVector = {1:[0,0], 0:[0,1], 2:[1,0], 3:[1,1]}
    sentiment   = {1:   0,    0:1,    2:1,     3:0}
    '''
    
    lengthInput = len(inputVector[0])

    # inputnodes have a value, weight
    # layernodes have a value, weight,
    # outputnodes only a value
    inputNodes = {'v':dict(), 'w':dict()}
    layerNodes = {'v':dict(), 'w':dict()}
    outputNodes = {'v':dict() }

    ###########################################################################################
    # load weights from file, if possible, else randomize weights    
    if os.path.exists(filename):
        try:
            weights = pickle.load( open('weights.txt', 'r') )
        except:
        
            print 'Randomized weights'
            for i in range( lengthInput + 1 ):
                inputNodes['w'][i] = dict()
                for j in range( num_hidden ):
                    # weight from node i to node j are 1
                    inputNodes['w'][i][j] = random.random()

            for j in range( num_hidden + 1 ):
                layerNodes['w'][j] = dict()
                for k in range( num_classes ):
                    # weight from node j to k are 1
                    layerNodes['w'][j][k] = random.random()
    
        print 'Loaded weights from file', filename
        number = 0
        for x in bagOfWords:
            # if a token exists in file weights
            if x in weights:
                inputNodes['w'][number]= dict()
                for j in range( num_hidden ):
                    # extract its weights to all hidden units 
                    inputNodes['w'][number][j] = weights[x][j]
            else:
                print 'Random weight for word ', x
                inputNodes['w'][number]= dict()
                for j in range( num_hidden ):
                    inputNodes['w'][number][j] = random.random()
            number += 1
        
        inputNodes['w'][lengthInput] = dict()
        for j in range( num_hidden ):
            # load bias weights
            inputNodes['w'][lengthInput][j] = weights['bias'][j]         

            layerNodes['w'][j] = dict()
            for k in range( num_classes ):
                layerNodes['w'][j][k] = weights['hidden'+str(j)][k]
        layerNodes['w'][num_hidden] = dict()
        for k in range( num_classes ):
            # load hidden bias weights
            layerNodes['w'][num_hidden][k] = weights['hidden'+str( num_hidden )][k]
    
    else:
        print 'Randomized weights'
        for i in range( lengthInput + 1 ):
            inputNodes['w'][i] = dict()
            for j in range( num_hidden ):
                # weight from node i to node j are 1
                inputNodes['w'][i][j] = random.random()

        for j in range( num_hidden + 1 ):
            layerNodes['w'][j] = dict()
            for k in range( num_classes ):
                # weight from node j to k are 1
                layerNodes['w'][j][k] = random.random()

    #####################################################################################
    
    for iteration in range( iterations ):
        print '\n:::::: Iteration', iteration + 1, 'of', iterations, '::::::'
        
        for t in trainSet:
            if not( t % 10 ):
                sys.stdout.write('.')
    
            # initialize input
            for i in range( lengthInput ):
                # copy input values
                inputNodes['v'][i] = inputVector[t][i]
            # bias input layer and hidden layer
            inputNodes['v'][lengthInput ] = -1
            layerNodes['v'][num_hidden  ]  = -1


            
            # forward progapagation
            for j in range( num_hidden ):
                inputValue = 0
                for i in range( lengthInput + 1 ):
                    inputValue += inputNodes['v'][i] * inputNodes['w'][i][j]
                    #print 'increment input by ', inputNodes['v'][i] ,'*', inputNodes['w'][i][j]
                #print 'Inputvalue in node 2,',j,'for input',inputVector[t],'  = ', inputValue
                
                layerNodes['v'][j] = (1  / (1 + math.exp( -inputValue )))  

                #print 'Hidden node',j,':  g(', inputValue, ')=', layerNodes['v'][j]
                
            for k in range( num_classes ):
                inputValue = 0
                for j in range( num_hidden + 1 ):
                    inputValue += layerNodes['v'][j] * layerNodes['w'][j][k]
                outputNodes['v'][k] = (1  / (1 + math.exp( -inputValue )))
                #print 'Output node',k,':  g(', inputValue, ')=', outputNodes['v'][k]

            
            # calculate error of each node delta: backward 
            delta = {1:dict(), 2:dict()}
            # neutral, desired output for sentiment == 0 is 1
            #delta[2][0] = sentiment[t] - outputNodes['v'][0]
            #print delta[2][0]
            delta[2][0] = outputNodes['v'][0] * (1-outputNodes['v'][0]) * ((sentiment[t]!=0) - outputNodes['v'][0])
            
            # positive
            #delta[2][1] = (sentiment[t] != 0)- outputNodes['v'][1]
            # negative
            #delta[2][2] = (sentiment[t] < 0) - outputNodes['v'][2]

            #print 'Sentiment = ', sentiment[t]
            #print '\nError layer 2:', delta[2]

            for j in range( num_hidden ):
                delta[1][j] = 0
                for k in range( num_classes ):
                    delta[1][j] += layerNodes['w'][j][k] * delta[2][k]
                delta[1][j] *= ( layerNodes['v'][j] * ( 1 - layerNodes['v'][j] ))

            #print 'Error layer 1', delta[1], '\n'

            # weight update i to j = alpha * delta j * output i
            
            alpha = 0.05 # learning rate
            
            for j in range( num_hidden + 1):
                for k in range( num_classes ):
                    layerNodes['w'][j][k] += alpha * layerNodes['v'][j] * delta[2][k]
                    
            for i in range( lengthInput + 1):
                for j in range( num_hidden):                    
                    inputNodes['w'][i][j] += alpha * inputNodes['v'][i] * delta[1][j]
                     
                    
    print '\n\nTime training took: ', time.time() - now

    ##########################################################################
    print 'Saving weights'
    if not(os.path.exists(filename)):
        #create a file
        open(filename, 'w')
    
    try:
        weights = pickle.load( open(filename, 'r') )
    except:
        weights = dict()
    i = 0
    for x in bagOfWords:
        weights[x] = dict()
        for j in range( num_hidden ):
            weights[x][j] = inputNodes['w'][i][j]
        i += 1
    
    weights['bias'] = dict()
    for j in range( num_hidden ):
        # save bias units
        weights['bias'][j] = inputNodes['w'][lengthInput][j]

        weights['hidden'+str(j)] = dict()
        for k in range( num_classes ):
            weights['hidden'+str(j)][k] = layerNodes['w'][j][k]

    weights['hidden'+str(num_hidden)] = dict()
    for k in range( num_classes ):
        weights['hidden'+str(num_hidden)][k]= layerNodes['w'][num_hidden][k]

        
    pickle.dump( weights, open( filename, 'w' ) )
    
    ###########################################################################
    confusion = {'tp':0,'fp':0,'tn':0,'fn':0}
    print '\nTesting..'
    
    for t in testSet:
        if not( t % 10 ):
            sys.stdout.write('.')
            
        s = sentence[t]
        # initialize input
        for i in range(lengthInput):
            # copy input values
            inputNodes['v'][i] = inputVector[t][i]
        inputNodes['v'][lengthInput] = -1
        layerNodes['v'][num_hidden ] = -1 

        # forward progapagation
        for j in range( num_hidden ):
            inputValue = 0
            for i in range( lengthInput + 1 ):
                inputValue += inputNodes['v'][i] * inputNodes['w'][i][j]
                #print 'increment input by ', inputNodes['v'][i] ,'*', inputNodes['w'][i][j]
                
            layerNodes['v'][j] = (1  / (1 + math.exp( -inputValue )))  
            #print 'Hidden node',j,':  g(', inputValue, ')=', layerNodes['v'][j]
            
        for k in range( num_classes ):
            inputValue = 0
            for j in range( num_hidden + 1):
                inputValue += layerNodes['v'][j] * layerNodes['w'][j][k]
            outputNodes['v'][k] = (1  / (1 + math.exp( -inputValue )))
            #print 'Output node',k,':  g(', inputValue, ')=', outputNodes['v'][k]
                
        #print 'True s:', sentiment[t], ',found', outputNodes['v'].values()
        #print '#',t, sentence[t],'--> output', outputNodes['v'][0]
        
        if (sentiment[t] != 0) and (outputNodes['v'][0] > 0.5):
            confusion['tp'] += 1
        elif (sentiment[t] != 0) and (outputNodes['v'][0] < 0.5):
            confusion['fn'] += 1
        elif (sentiment[t] == 0) and (outputNodes['v'][0] < 0.5):
            confusion['tn'] += 1
        else:
            confusion['fp'] += 1
    print ''
    print confusion

    raw_input('Press enter to exit')
    
        

def createWordVectors(trainSet, num_sentences, sentence ):
    # Create the bag of words
    print 'Creating the bag of words'
    bagOfWords = list()

    global weights
    
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
        vec = [ tk_sentence.count(x) for x in bagOfWords ]
        
        wordVectors[i] = vec

    return wordVectors, bagOfWords

neuralNetwork()
