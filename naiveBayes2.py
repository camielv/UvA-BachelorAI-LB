import csv
import perceptron
import random
import time
import re

class Main():

    # Open a file
    file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')     

    # Initialize dictionaries
    sentence = {}
    sentiment = {}
    corpus = {}
    probSent = {}
    
    # Initialize lists
    trainSet = []
    testSet = []

    # The number of sentences
    num_sentences = 0

<<<<<<< HEAD
    def __init__( self, iterations = 2, total_messages = 10000 ):
=======
    def __init__( self, iterations = 4, total_messages = 10000 ):
>>>>>>> f4a018cb96f181d0bf5a98d02ca9539281604a38
               
        # Get current time
        now = time.time()

        # The n for the n-grams
        self.n = 4

        # Number of classes and distribution
<<<<<<< HEAD
        self.class_distribution = [0],[-1,1,-2,2]
=======
        self.class_distribution = [2],[-2],[-1],[1],[0]
>>>>>>> f4a018cb96f181d0bf5a98d02ca9539281604a38
        self.num_classes = len( self.class_distribution )
        
        # Load the sentences and sentiments from file
        self.initializeCorpus( total_messages )
        
        # Create confusion matrix
        allconfusion = [ [ 0 for x in range( self.num_classes ) ] for y in range( self.num_classes ) ]
            
        for i in range( iterations ):
            print "--- iteration", i + 1, "of", iterations, "---"
            
            # Reset variables
            self.corpus = {}
            self.probSent = {}
            self.trainSet = []
            self.testSet = []


            # Random selection of training and test data
            self.makeCorpus( distribution = (0.7, 0.3) )
            self.testMethod()
            
            # Create test set
            temp_set = [],[]
            for j in self.testSet:
<<<<<<< HEAD
                
=======
                m = max( self.probSent[j] )

>>>>>>> f4a018cb96f181d0bf5a98d02ca9539281604a38
                # Find the correct class
                for c in range( self.num_classes ):
                    if self.sentiment[j] in self.class_distribution[c]:
                        temp_set[1].append( c )
                        break

                # Find the classified class
<<<<<<< HEAD
                m = self.probSent[j].index( max( self.probSent[j] ) )
                temp_set[0].append( m )
=======
                for c in range( self.num_classes ):
                    if m == self.probSent[j][c]:
                        temp_set[0].append( c )
                        break
>>>>>>> f4a018cb96f181d0bf5a98d02ca9539281604a38
                    
            # Create confusion matrix
            confusion = [ [ 0 for x in range( self.num_classes ) ] for y in range( self.num_classes ) ]

            # Fill confusion matrix
            for j in range( len( temp_set[0] ) ):
                clas = temp_set[0][j]
                sent = temp_set[1][j]

                confusion[sent][clas] += 1
                
            print confusion
            for j in range( self.num_classes ):
                for k in range( self.num_classes ):
                    allconfusion[j][k] += confusion[j][k]

        # Calculate mean
        print 'Average:'
        total = 0.0
        for i in range( self.num_classes ):
            for j in range( self.num_classes ):
                allconfusion[i][j] /= float( iterations )
                total += allconfusion[i][j]

        #acc = (allconfusion[0][0] + allconfusion[1][1] + allconfusion[2][2]) / float(total)

        print '\t\tRecall\t\tPrecision\tAccuracy'
        for i in range( self.num_classes ):
            row_sum = sum( allconfusion[i] )
            col_sum = 0
            for j in range( self.num_classes ):
<<<<<<< HEAD
                col_sum += allconfusion[j][i]
=======
                col_sum += confusion[j][i]
>>>>>>> f4a018cb96f181d0bf5a98d02ca9539281604a38
                
            truepositives = float(allconfusion[i][i])
            truenegatives = total - row_sum - col_sum + truepositives
            falsepositives = col_sum - truepositives
            falsenegatives = row_sum - truepositives
            
            rec = truepositives / (truepositives + falsenegatives )
            pre = truepositives / ( truepositives + falsepositives )
            acc = (truepositives + truenegatives) / total
            print 'Class {0}:\t{1}\t{2}\t{3}'.format( i, rec, pre, acc )

        print '\n', allconfusion

        print 'Time taken for', iterations, 'iterations: ', time.time()- now
            
    '''
        Corpus methods
    '''        
    def initializeCorpus( self, max_num = 10000, tweet_only = True ):
        self.sentence = {}
        self.sentiment = {}

        # Initialize counter
        i = 0

        print 'Creating corpus with ', self.n , '- grams.'

        # Collect sentences and sentiments
        for entry in self.file1:
            # Do not include header
            if i == 0:
                i+=1
                continue

            # Check for tweets
            if tweet_only:
                if int( entry[3] ) != 3:
                    continue
            
            # The actual message is the 9th attribute, sentiment is the 4th
            curSent = re.sub('\||#|:|;|RT|@\w+|\**', '', entry[9])
            sent = float( entry[4] )

            self.sentence[i - 1] = curSent
            self.sentiment[i - 1] = sent
            # Stop at 10000
            i += 1
            if ( i == max_num ):
                break
        # Set the number of sentences
        self.num_sentences = i
        print 'Number of sentences =', self.num_sentences
        
    def tokenize( self, sentence ):  
        return re.findall( '\w+|\?|\!', sentence )

    def clean( self, sentence ):
        # print sentence
        sentence = sentence.replace( ':-)', " blijesmiley " )
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( ':s', ' awkwardsmiley ' )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )
        
        # delete non-expressive words
        #sentence = re.sub('en|de|het|ik|jij|zij|wij|deze|dit|die|dat|is|je|na|zijn|uit|tot|te|sl|hierin|naar|onder', '', sentence)
        
        # Delete expressions, such as links, hashtags, twitteraccountnames 
        sentence = re.sub( r'\:P|\:p|http\/\/t\.co\/\w+|\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub( ' +',' ', sentence )
        
        # remove double letters
        #for x in 'abcdefghijklmnopqrstuvwxyz':
        #    sentence = re.sub(x+'+', x, sentence )

        return sentence

            
    def makeCorpus( self, distribution ):
        # Create corpus and count word frequencies
        
        print 'Splitting corpus'
        for i in range( 1, self.num_sentences ):
            # Assign at random to train, test or validation set
            r = random.random()
            if ( r < distribution[0] ):
                self.trainSet.append(i-1)
            else:
                self.testSet.append(i-1)
    
        for i in self.trainSet:
            # Tokenize the sentence
            tk_sent = self.tokenize( self.clean( self.sentence[i] ) )
       
            # Create temporary dictionary of dictionaries of lists
            temp_ngram = {}

            for k in range( 1, self.n + 1 ):
                temp_ngram[k] = {}
                for j in range( 1, k + 1 ):
                    temp_ngram[k][j] = []

            count = 0;
            # Iterate over every word
            for word in tk_sent:
                count += 1
                # Loop over every n-gram
                for k in range( 1, self.n + 1 ):
                    # Loop over every temporary instantion of an n gram
                    for j in range( 1, k + 1 ):
                        # Add this word
                        if count >= j:
                            temp_ngram[k][j].append( word )
                        
                        if len( temp_ngram[k][j] ) == k:
                            # We found a n-gram
                            token = tuple( temp_ngram[k][j] )

                            # format: corpus[<combination of n tokens>]{neutrals, positives, negatives}
                            for c in range( self.num_classes ):
                                # Find out which class it is in
                                if self.sentiment[i] in self.class_distribution[c]:
                                    if token in self.corpus:
                                        self.corpus[token][c] += 1
                                    else:
                                        self.corpus[token] = [ 0 for x in range( self.num_classes ) ]
                                        self.corpus[token][c] = 1
                                    break
                                    
                            temp_ngram[k][j] = []
                            
    def testMethod(self):
        print 'Testing'
        
        # Calculate prior probabilities
        sums = [ 0 for c in range( self.num_classes ) ]
            
        for token in self.corpus.keys():
            for c in range( self.num_classes ):
                sums[c] += self.corpus[token][c]

        totalSum = sum( sums )
        priors = [ (float( sums[c] ) / totalSum) for c in range( self.num_classes ) ]

        # Calculate probabilities for every sentence
        for i in self.testSet:
            tk_sent = self.tokenize( self.clean( self.sentence[i] ) )
            # Create temporary dictionary of dictionaries of lists
            temp_ngram = {}

            for k in range( 1, self.n + 1 ):
                temp_ngram[k] = {}
                for j in range( 1, k + 1 ):
                    temp_ngram[k][j] = []

            # Counter used for ngram finding
            count = 0;
            
            # Running probability totals
            classes = [ priors[c] for c in range( self.num_classes ) ]
            
            # Iterate over every word
            for word in tk_sent:
                count += 1
                # Loop over every n-gram
                for k in range( 1, self.n + 1 ):
                    # Loop over every temporary instantion of an n gram
                    for j in range( 1, k + 1 ):
                        # Add this word
                        if count >= j:
                            temp_ngram[k][j].append( word )
                        
                        if len( temp_ngram[k][j] ) == k:
                            # We found a n-gram
                            token = tuple( temp_ngram[k][j] )

                            # Check if the token is in the corpus
                            if token in self.corpus:
                                for c in range( self.num_classes ):
                                    # If so, add the chances to the probabilities
                                    classes[c] *= float( self.corpus[token][c] ) / sum( self.corpus[token] )
                                    
                            temp_ngram[k][j] = []    

            self.probSent[i] = classes
            
    
m = Main()
