import re
import cPickle as pickle
import math

def clean( sentence ):
    sentence = sentence.lower()
    sentence = sentence.replace( ':-)', " blijesmiley " )
    sentence = sentence.replace( ':)', " blijesmiley " )
    sentence = sentence.replace( ':(', " zieligesmiley " )
    sentence = sentence.replace( ':s', ' awkwardsmiley ' )
    sentence = sentence.replace( '!', " ! " )
    sentence = sentence.replace( '?', " ? " )
    
    # Delete useless info, such as links, hashtags, twitteraccountnames 
    sentence = re.sub('rt |@\w+|http.*', '', sentence)
    sentence = re.sub( r'\.|\,|\/|\\', '', sentence )
    sentence = re.sub( r'\[|\]|&#39;|\||#|:|;|\(|\)|\**', '', sentence )
    sentence = re.sub( ' +',' ', sentence )

    # delete non-expressive words
    for x in ['bij','!','of','voor','in','een','he','op','wie','uit','eo','en','de','het','ik','jij','zij','wij','deze','dit','die','dat','is','je','na','zijn','uit','tot','te','sl','hierin','naar','onder','is']:
        sentence = re.sub(' '+x+' ',' ',sentence)
        sentence = re.sub('\A'+x+' ',' ',sentence)
        sentence = re.sub(' '+x+'\Z', ' ', sentence)
    return sentence
    # print sentence
    # Werkt nog niet cleanup is nog niet goed genoeg
    #return __stemmer.stem( sentence )

def tokenize( sentence ):
    return re.findall('\w+|\?|\!', sentence)

def classifyNewLine(sentence, filename = './weightsDayTraining.txt'):
    
    # Change format for sentence
    sentence = tokenize( clean( sentence ))

    num_hidden = 5
    num_classes = 1
    
    # inputnodes have a value, weight
    # layernodes have a value, weight,
    # outputnodes only a value
    inputNodes = {'v':dict(), 'w':dict()}
    layerNodes = {'v':dict(), 'w':dict()}
    outputNodes = {'v':dict() }

    
    ###########################################################################################

    # Step 1, initialize weights
    
    # load weights from file, if possible, else randomize weights    
    weights = pickle.load( open(filename, 'r') )    

    # remove hidden, bias
    print 'Loading weights from file', filename
    number = 0
    # use all tokens as a bag of words to form a word vector containing all possible features
    for x in weights:
        # and if not a hidden node weight or a bias unit
        if not(re.match('hidden.*|bias',x)):
            inputNodes['w'][number]= dict()
            for j in range( num_hidden ):
                # extract its weights to all hidden units 
                inputNodes['w'][number][j] = weights[x][j]
            number += 1
    
    # the n+1th unit is the bias node
    inputNodes['w'][ number ] = dict()
    for j in range( num_hidden ):
        # load bias weights
        inputNodes['w'][ number ][j] = weights['bias'][j]         

    
        layerNodes['w'][j] = dict()
        for k in range( num_classes ):
            layerNodes['w'][j][k] = weights['hidden'+str(j)][k]
    layerNodes['w'][num_hidden] = dict()
    for k in range( num_classes ):
        # load hidden bias weights
        layerNodes['w'][num_hidden][k] = weights['hidden'+str( num_hidden )][k]
    #print inputNodes
    #print layerNodes
    #####################################################################################

    # Step 2: create wordVector from weights dictionary
    wordVector = list()
    for x in weights.keys():
        # skip all hidden and bias 
        if not(re.match('hidden.*|bias',x)):
            if x in sentence:
                wordVector.append(1)
            else:
                wordVector.append(0)
    #print wordVector
    #####################################################################################
    
    # Step 3: forward propagate input through this network to retrieve output
    
    # initialize input
    for i in range( len( wordVector ) ):
        # copy input values
        inputNodes['v'][i] = wordVector[i]
    # bias input layer and hidden layer
    inputNodes['v'][ len(wordVector) ] = -1
    layerNodes['v'][num_hidden  ]  = -1
    
    # forward progapagation
    for j in range( num_hidden ):
        inputValue = 0
        for i in range( len( wordVector ) + 1 ):
            inputValue += inputNodes['v'][i] * inputNodes['w'][i][j]
        
        layerNodes['v'][j] = (1  / (1 + math.exp( -inputValue )))  

        #print 'Hidden node',j,':  g(', inputValue, ')=', layerNodes['v'][j]
        
    for k in range( num_classes ):
        inputValue = 0
        for j in range( num_hidden + 1 ):
            inputValue += layerNodes['v'][j] * layerNodes['w'][j][k]
        outputNodes['v'][k] = (1  / (1 + math.exp( -inputValue )))

        #print 'Output node',k,':  g(', inputValue, ')=', outputNodes['v'][k]
    return outputNodes['v'][0]
    
