# Data Analyser
import csv
import types
import sys
import operator
import re

class DataAnalyser():
    __neutral = dict()
    __positive = dict()
    __negative = dict()

    def __init__( self, dataset, n ):
        __neutral = __positive = __negative = dict()
        self.__analyse( dataset, n )

    def __analyse( self, dataset, n ):
        for entry in dataset:
            sentence = self.__tokenize( self.__clean( entry[9] ) )
            try:
                sentiment = int( entry[4] )
            except ValueError:
                continue

            for i in range( 1, len( sentence ) - ( n - 1 ) ):
                token = tuple(sentence[i-1 : (i-1)+n])
                if( sentiment == 0 ):
                    if( self.__neutral.has_key(token) ):
                        self.__neutral[token] += 1
                    else:
                        self.__neutral[token] = 1
                elif( sentiment > 0 ):
                    if( self.__positive.has_key(token) ):
                        self.__positive[token] += 1
                    else:
                        self.__positive[token] = 1
                else:
                    if( self.__negative.has_key(token) ):
                        self.__negative[token] += 1
                    else:
                        self.__negative[token] = 1

    def __clean( self, sentence ):
        #print sentence
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

        # delete non-expressive words
        wordlist = ['bij', 'in', 'van', 'een', 'he','op','wie','uit','eo','en','de','het','ik','jij','zij','wij','deze','dit','die','dat','is','je','na','zijn','uit','tot','te','sl','hierin','naar','onder','is']
        for x in wordlist:
            sentence = re.sub(' '+x+' ',' ', sentence)
            sentence = re.sub('\A'+x+' ',' ', sentence)
            sentence = re.sub(' '+x+'\Z',' ', sentence)
        #print sentence
        #raw_input('Press enter')

        return sentence

    def __tokenize( self, sentence ):
        return re.findall('\w+|\?', sentence)

    def saveFile(self, filename):
        print "Saving to file \"" + filename + "\"..."
        doc = open(filename, "w")
        doc.write("...ANALYSIS OF DATASET...\n\n")

        tokens_neutral = sum( self.__neutral.values() )
        tokens_positive = sum( self.__positive.values() )
        tokens_negative = sum( self.__negative.values() )

        diff_neutral = len( self.__neutral )
        diff_positive = len( self.__positive )
        diff_negative = len( self.__negative )

        doc.write( "...All Tokens...\n" )
        doc.write( "Total tokens: " + str(tokens_neutral + tokens_positive + tokens_negative) + "\n" )
        doc.write( "Neutral tokens: " + str(tokens_neutral) + "\n" )
        doc.write( "Positive tokens: " + str(tokens_negative) + "\n" )
        doc.write( "Negative tokens: " + str(tokens_negative) + "\n\n" )

        doc.write( "...Different Tokens...\n" )
        doc.write( "Total tokens: " + str(diff_neutral + diff_positive + diff_negative) + "\n" )
        doc.write( "Neutral tokens: " + str(diff_neutral) + "\n" )
        doc.write( "Positive tokens: " + str(diff_negative) + "\n" )
        doc.write( "Negative tokens: " + str(diff_negative) + "\n\n" )
        doc.write( "...Results...\n" )
        sort_neutral = sorted(self.__neutral.iteritems(), key=operator.itemgetter(1), reverse = True)
        sort_positive = sorted(self.__positive.iteritems(), key=operator.itemgetter(1), reverse = True)
        sort_negative = sorted(self.__negative.iteritems(), key=operator.itemgetter(1), reverse = True)

        for i in range( len( sort_neutral) ):
            doc.write( "...#" + str(i + 1) + "...\n" )
            if( i < len(sort_neutral) ):
                doc.write( "Neutral: " + str(sort_neutral[i]) + " Percentage: " + str(sort_neutral[i][1] / float(tokens_neutral)) + "\n")
            if( i < len(sort_positive) ):
                doc.write( "Positive: " + str(sort_positive[i]) + " Percentage: " + str(sort_positive[i][1] / float(tokens_positive)) + "\n")
            if( i < len(sort_negative) ):
                doc.write( "Negative: " + str(sort_negative[i]) + " Percentage: " + str(sort_negative[i][1] / float(tokens_negative)) + "\n")
            doc.write( "\n" ) 
        doc.close()
        print "Done..."

    def printit(self, iterations):
        tokens_neutral = sum( self.__neutral.values() )
        tokens_positive = sum( self.__positive.values() )
        tokens_negative = sum( self.__negative.values() )

        diff_neutral = len( self.__neutral )
        diff_positive = len( self.__positive )
        diff_negative = len( self.__negative )
        
        print "All Tokens ---"
        print "Total tokens...", (tokens_neutral + tokens_positive + tokens_negative)
        print "Neutral tokens...", tokens_neutral
        print "Positive tokens...", tokens_negative
        print "Negative tokens...", tokens_negative, "\n"

        print "Different Tokens ---"
        print "Total tokens...", (diff_neutral + diff_positive + diff_negative)
        print "Neutral tokens...", diff_neutral
        print "Positive tokens...", diff_negative
        print "Negative tokens...", diff_negative, "\n"

        sort_neutral = sorted(self.__neutral.iteritems(), key=operator.itemgetter(1), reverse = True)
        sort_positive = sorted(self.__positive.iteritems(), key=operator.itemgetter(1), reverse = True)
        sort_negative = sorted(self.__negative.iteritems(), key=operator.itemgetter(1), reverse = True)

        for i in range(iterations):
            print '#' + str(i + 1) + '...' 
            if( i < len(sort_neutral) ):
                print 'Neutral: ', sort_neutral[i]
            if( i < len(sort_positive) ):
                print 'Positive: ', sort_positive[i]
            if( i < len(sort_negative) ):
                print 'Negative: ', sort_negative[i]
            print "" 

dataset = "../DataCSV.csv"
n = 1
filename = "Analysis.txt"
args = sys.argv

if ( len(args) > 3 ):
    print "Succes!"
    dataset = args[1]
    filename = args[2]
    try:
        n = int(args[3])
    except ValueError:
        pass

dataset = csv.reader( open( dataset , 'rb' ), delimiter=',', quotechar='"' )  
analysis = DataAnalyser( dataset, n )
analysis.saveFile( filename )
