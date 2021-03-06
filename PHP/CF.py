# Filename: CF.py
# Written by: Camiel Verschoor
# Definition: Classifier

import cPickle as pickle
import re
import math

class Classifier():
    # Number of classes and distribution
    __class_distribution = [0],[-1,-2],[1,2]
    __display_class = 1,2
    __num_classes = len( __class_distribution )
    __corpus = pickle.load( open('weightsDayTraining.txt') )
    __n = 4
    __weights = pickle.load( open('.txt') )
    __sentiment = None

    def __init__( self ):
        pass

    def __tokenize( self, sentence ):  
        return re.findall( '\w+|\?|\!', sentence )

    def __clean( self, sentence ):
        sentence = sentence.lower()
        sentence = sentence.replace( ':-)', " blijesmiley " )
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( ':s', ' awkwardsmiley ' )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )

        # Delete useless info, such as links, hashtags, twitteraccountnames 
        sentence = re.sub('rt |@\w+|http.*', '', sentence )
        sentence = re.sub( r'\.|\,|\/|\\', '', sentence )
        sentence = re.sub( r'\[|\]|&#39;s|\||#|:|;|\(|\)|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )

        # Delete non-expressive words
        wordlist = ['bij', 'in', 'van', 'een', 'he', 'op', 'wie', 'uit', 'eo', 'en', 'de', 'het', 'ik', 'jij', 'zij', 'wij', 'deze', 'dit', 'die', 'dat', 'is', 'je', 'na', 'zijn', 'uit', 'tot', 'te', 'sl', 'hierin', 'naar', 'onder', 'is']
        for x in wordlist:
            sentence = re.sub(' '+x+' ',' ', sentence)
            sentence = re.sub('\A'+x+' ',' ', sentence)
            sentence = re.sub(' '+x+'\Z',' ', sentence)

        return sentence

    def classify( self, sentence ):
        sentence = self.__tokenize( self.__clean( sentence ) )
        (success, P_negative, P_positive) = self.__weightedProbabilitySum( sentence )
        P_opinion = self.__neuralNetwork( sentence )
        
        certainty_opinion = abs( P_opinion - 0.5 ) * 2
        certainty_negpos = abs( P_positive - P_negative ) * math.sqrt(2)
        
        if( P_opinion > 0.5 ):                
            if( P_negative > P_positive):
                return (success, -1, P_negative, certainty_negpos)
            else:
                return (success, 1, P_positive, certainty_negpos)
        else:
            return (success, 0, P_opinion, certainty_opinion)

    ### ALGORITHMS ###

    ## WEIGHTED PROBABLIITY SUM ##
    def __weightedProbabilitySum( self, tk_sent ):
        # Create temporary dictionary of dictionaries of lists
        temp_ngram = {}

        for k in range( 1, self.__n + 1 ):
            temp_ngram[k] = {}
            for j in range( 1, k + 1 ):
                temp_ngram[k][j] = []

        # Counter used for ngram finding
        count = 0

        # Number of found features
        features = 0

        # Running probability totals
        classes = [ 0 for c in range( self.__num_classes ) ]

        # Iterate over every word
        for word in tk_sent:
            count += 1
            # Loop over every n-gram
            for k in range( 1, self.__n + 1 ):
                # Loop over every temporary instantion of an n gram
                for j in range( 1, k + 1 ):
                    # Add this word
                    if count >= j:
                        temp_ngram[k][j].append( word )
                    
                    if len( temp_ngram[k][j] ) == k:
                        # We found a n-gram
                        token = tuple( temp_ngram[k][j] )

                        # Check if the token is in the corpus
                        if token in self.__corpus:
                            for c in range( self.__num_classes ):
                                # If so, add the chances to the probabilities
                                classes[c] += float( self.__corpus[token][c] ) / sum( self.__corpus[token] )
                                features += 1
                                
                        temp_ngram[k][j] = []
                    # Normalize to the number of features
        success = True
        try:
            for c in range( self.__num_classes ):
                classes[c] /= float( features )
        except:
            success = False

        # Return no. features, P_neg, P_pos
        return ( success, classes[1], classes[2] )

    ## NEURAL NETWORK ##
    def __neuralNetwork( self, sentence ):
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
        
        # remove hidden, bias
        number = 0
        # use all tokens as a bag of words to form a word vector containing all possible features
        for x in self.__weights:
            # and if not a hidden node weight or a bias unit
            if not(re.match('hidden.*|bias',x)):
                inputNodes['w'][number]= dict()
                for j in range( num_hidden ):
                    # extract its weights to all hidden units 
                    inputNodes['w'][number][j] = self.__weights[x][j]
                number += 1
        
        # the n+1th unit is the bias node
        inputNodes['w'][ number ] = dict()
        for j in range( num_hidden ):
            # load bias weights
            inputNodes['w'][ number ][j] = self.__weights['bias'][j]         

        
            layerNodes['w'][j] = dict()
            for k in range( num_classes ):
                layerNodes['w'][j][k] = self.__weights['hidden'+str(j)][k]
        layerNodes['w'][num_hidden] = dict()
        for k in range( num_classes ):
            # load hidden bias weights
            layerNodes['w'][num_hidden][k] = self.__weights['hidden'+str( num_hidden )][k]

        #####################################################################################

        # Step 2: create wordVector from weights dictionary
        wordVector = list()
        for x in self.__weights.keys():
            # skip all hidden and bias 
            if not(re.match('hidden.*|bias',x)):
                if x in sentence:
                    wordVector.append(1)
                else:
                    wordVector.append(0)

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

        for k in range( num_classes ):
            inputValue = 0
            for j in range( num_hidden + 1 ):
                inputValue += layerNodes['v'][j] * layerNodes['w'][j][k]
            outputNodes['v'][k] = (1  / (1 + math.exp( -inputValue )))

            #print 'Output node',k,':  g(', inputValue, ')=', outputNodes['v'][k]
        return outputNodes['v'][0]
