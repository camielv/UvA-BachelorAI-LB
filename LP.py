# Language Processor
import nltk
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
    __bagOfWords = dict()
    __emptyBag = dict()
    __num_sentences = 0

    __stemmer = nltk.stem.SnowballStemmer("dutch")

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
            curSent = self.__tokenize( self.__clean(entry[9]))
            sent = float(entry[4])
            
            self.__sentence[i - 1] = curSent
            self.__sentiment[i - 1] = sent

            # init for bag of n words

            
            # Iterate over every n tokens
            for j in range(len( curSent )-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(curSent[j:j+n])

                
                self.__emptyBag[token] = 0 
                # format: corpus[<combination of n tokens>] = {neutrals, positives, negatives}
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

    def makeBagOfWords(self, n=3, distribution = (0.7, 0.3) ):
        self.__trainSet = list()
        self.__testSet = list()

        for i in range( 1, self.__num_sentences ):

            # initialize bagofwords
            self.__bagOfWords[i] = self.__emptyBag
            # Assign at random to train, test or validation set
            r = random.random()
            if ( r < distribution[0] ):
                self.__trainSet.append(i-1)
            else:
                self.__testSet.append(i-1)

        for i in self.__trainSet:
            tk_sent = self.__sentence[i]
            for j in range(len(tk_sent) - (n-1)):
                token = tuple(tk_sent[j:j+n])
                bagOfWords[i][token] = 1

        
    def makeCorpus(self, n = 3, distribution = (0.7, 0.3 ) ):
        self.__trainSet = list()
        self.__testSet = list()
        self.__probWord = {'Opinion':dict(), 'PosNeg':dict()}
        self.__probSent = {'Opinion':dict(), 'PosNeg':dict()}

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
            tk_sent = self.__sentence[i]
            pOpinion = 0
            pPosNeg  = 0

            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                self.__probWord['Opinion'][token] = float( self.__corpus[token][1] + self.__corpus[token][2]) / self.__corpus[token][0]
                pOpinion = pOpinion + self.__probWord['Opinion'][token]

                # if token contains sentiment, sentence may contain sentiment as well
                if self.__corpus[token][1] or self.__corpus[token][2]:
                    self.__probWord['PosNeg'][token]  = float( self.__corpus[token][1] ) / ( self.__corpus[token][1] + self.__corpus[token][2] )
                    pPosNeg = pPosNeg + self.__probWord['PosNeg'][token]
            try:    
                self.__probSent['Opinion'][i] = pOpinion / float(len(tk_sent)-2) # to be extra certain intdiv does not occur
                self.__probSent['PosNeg'][i]  = pPosNeg  / float(len(tk_sent)-2)
        
            except:
                print self.__sentence[i]
                self.__probSent['Opinion'][i] = pOpinion  # to be extra certain intdiv does not occur
                self.__probSent['PosNeg'][i]  = pPosNeg  
        
            
        # Calculate probabilities for testset
        for i in self.__testSet:
            tk_sent = self.__sentence[i]
            pOpinion = 0
            pPosNeg  = 0
            
            # Iterate over every n tokens
            for j in range(len(tk_sent)-(n-1)):
                # token is now a uni/bi/tri/n-gram instead of a token
                token = tuple(tk_sent[j:j+n])
                try:
                    pOpinion = pOpinion + self.__probWord['Opinion'][token]
                except:
                    # If word does not occur in corpus, ignore for now
                    # (can try katz backoff later?)
                    pass
                try:
                    pPosNeg  = pPosNeg  + self.__probWord['PosNeg'][token]
                except:
                    pass
            
            # Store the probability in dictionary
            try:
                self.__probSent['Opinion'][i] = pOpinion / float(len(tk_sent)-2) # to be extra certain intdiv does not occur
                self.__probSent['PosNeg'][i] = pPosNeg   / float(len(tk_sent)-2)
            except:
                print self.__sentence[i]
                self.__probSent['Opinion'][i] = pOpinion # to be extra certain intdiv does not occur
                self.__probSent['PosNeg'][i] = pPosNeg   
                
        return (self.__trainSet, self.__testSet, self.__probWord, self.__probSent)

    def __clean( self, sentence ):
        # print sentenc
        sentence = sentence.replace( ':-)', " blijesmiley " )
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( ':-(', " zieligesmiley " )
        sentence = sentence.replace( ':s', ' awkwardsmiley ' )
        sentence = sentence.replace( ':-s', ' awkwardsmiley ' )
        sentence = sentence.replace( ':p', ' tongsmiley ' )
        sentence = sentence.replace( ':-p', ' tongsmiley ' )
        sentence = sentence.replace( ':@', ' angrysmiley ' )
        sentence = sentence.replace( ':-@', ' angrysmiley ' )
        sentence = sentence.replace( '(6)', ' devilsmiley ' )
        sentence = sentence.replace( '(a)', ' angelsmiley ' )
        sentence = sentence.replace( '(A)', ' angelsmiley ' )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )

        # Delete expressions, such as links, hashtags, twitteraccountnames 
        sentence = re.sub( r'http\/\/t\.co\/\w+|\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )
        return sentence
        # print sentence
        # Werkt nog niet cleanup is nog niet goed genoeg
        return self.__stemmer.stem( sentence )

    def __tokenize( self, sentence ):
        return re.findall('\w+|\?|\!', sentence)

    # Get Functions
    def getCorpus( self ):
        return self.__corpus

    def getSentence( self ):
        return self.__sentence

    def getSentiment( self ):
        return self.__sentiment

    def getNum_sentences( self ):
        return self.__num_sentences
