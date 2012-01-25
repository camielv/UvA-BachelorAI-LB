# Language Processor
import re
import random

class LanguageProcessor():
    # Variables
    __trainSet = list()
    __testSet = list()

    __corpus = dict()
    __sentence = dict()
    __sentiment = dict()
    __probWord = dict()
    __probSent = dict()

    __num_sentences = 0

    def __init__(self, file1, n = 3, max_num = 10000, tweet_only = True):
        self.__corpus = dict()
        self.__sentence = dict()
        self.__sentiment = dict()
        self.__initializeCorpus(file1, n, max_num, tweet_only)

    def __initializeCorpus( self, file1, n, max_num = 10000, tweet_only = True ):
        # Initialize counter
        i = 0
        print 'Creating corpus with ', n , '- grams.'

        # Collect sentences and sentiments
        for entry in file1:
            # Do not include header
            if i == 0:
                i+=1
                continue

            # Check for tweets
            if tweet_only:
                if int( entry[3] ) != 3:
                    continue

            # The actual message is the 9th attribute, sentiment is the 4th
            curSent = self.__clean(entry[9])
            sent = float(entry[4])
            
            self.__sentence[i - 1] = curSent
            self.__sentiment[i - 1] = sent

            # Tokenize the sentence
            tk_sent = self.__tokenize( curSent )
            
            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                
                # format: corpus[<combination of n tokens>]{neutrals, positives, negatives}
                if token in self.__corpus:
                    if sent > 0:
                        self.__corpus[token] = self.__corpus[token][0] + 1, self.__corpus[token][1] + 1, self.__corpus[token][2]
                    elif sent == 0:
                        self.__corpus[token] = self.__corpus[token][0] + 1, self.__corpus[token][1], self.__corpus[token][2]
                    else:
                        self.__corpus[token] = self.__corpus[token][0] + 1, self.__corpus[token][1], self.__corpus[token][2] + 1
                else:
                    if sent > 0:
                        self.__corpus[token] = 1, 1, 0
                    elif sent == 0:
                        self.__corpus[token] = 1, 0, 0
                    else:
                        self.__corpus[token] = 1, 0, 1
            # Stop at 10000
            i += 1
            if ( i == max_num ):
                break
        # Set the number of sentences
        self.__num_sentences = i
        print 'Number of sentences =', self.__num_sentences

    def makeCorpus(self, n, distribution):
        self.__trainSet = list()
        self.__testSet = list()
        self.__probWord = dict()
        self.__probSent = dict()

        for i in range( 1, self.__num_sentences ):
            # Assign at random to train, test or validation set
            r = random.random()
            if ( r < distribution[0] ):
                self.__trainSet.append(i-1)
            else:
                self.__testSet.append(i-1)
            
        print 'Calculating n-gram probability'
        # Corpus created, calculate words probability of sentiment based on frequency
        for i in self.__trainSet:
            tk_sent = self.__tokenize( self.__sentence[i] )
            p = 0

            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                self.__probWord[token] = float(self.__corpus[token][1]) / self.__corpus[token][0]
                # print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'
                p = p + self.__probWord[token]
            self.__probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
        
        return (self.__trainSet, self.__testSet, self.__probWord, self.__probSent)

    def __clean( self, sentence ):
        sentence = sentence.replace( ':-)', " blijesmiley " )
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )

        # Delete expressions
        sentence = re.sub( r'\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        return re.sub( ' +',' ', sentence )

    def __tokenize( self, sentence ):
        return sentence.split( ' ' )

    # Get Functions
    def getCorpus( self ):
        return self.__corpus

    def getSentence( self ):
        return self.__sentence

    def getSentiment( self ):
        return self.__sentiment
