# Language Processor
import re
import random

class LanguageProcessor():
    corpus = dict()
    sentence = dict()
    sentiment = dict()

    num_sentences = 0

    def __init__(self, file1, n = 3, max_num = 10000, tweet_only = True):
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
            curSent = entry[9]
            sent = float(entry[4])
            
            self.sentence[i - 1] = curSent
            self.sentiment[i - 1] = sent
            
            # Tokenize the sentence
            tk_sent = self.tokenize( curSent )
            
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

    def tokenize( self, sentence ):
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )

        # Delete expressions
        sentence = re.sub( r'\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )

        return sentence.split( ' ' )
'''        
    def makeCorpus(self, sentence, n, corpus distribution = {0.7, 0.3}):
        trainSet = list()
        testSet = list()
        probWord = dict()
        for i in range( 1, num_sentences ):
            # Assign at random to train, test or validation set
            r = random.random()
            if ( r < distribution[0] ):
                trainSet.append(i-1)
            else:
                testSet.append(i-1)
            
        print 'Calculating unigram probability'
        probWord = {}
        # Corpus created, calculate words probability of sentiment based on frequency
        for i in trainSet:
            tk_sent = self.tokenize( sentence[i] )
            p = 0

            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                probWord[token] = float(self.corpus[token][1]) / self.corpus[token][0]
                # print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'
                p = p + self.probWord[token]
            self.probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
    def tokenize( self, sentence ):
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )

        # Delete expressions
        sentence = re.sub( r'\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )

    def tokenize( self, sentence ):
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )

        # Delete expressions
        sentence = re.sub( r'\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )

        return sentence.split( ' ' )
'''
