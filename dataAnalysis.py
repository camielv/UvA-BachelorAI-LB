# Data Analyser
import csv
import operator
import re

class DataAnalyser():
    __neutral = dict()
    __positive = dict()
    __negative = dict()

    def __init__( self, dataset ):
        __neutral = __positive = __negative = dict()
        self.__analyse( dataset )

    def __analyse( self, dataset ):
        for entry in dataset:
            sentence = self.__tokenize( self.__clean( entry[9] ) )
            try:
                sentiment = int( entry[4] )
            except ValueError:
                continue

            for token in sentence:
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
        sentence = sentence.replace( ':-)', " blijesmiley " )
        sentence = sentence.replace( ':)', " blijesmiley " )
        sentence = sentence.replace( ':(', " zieligesmiley " )
        sentence = sentence.replace( ':s', ' awkwardsmiley ' )
        sentence = sentence.replace( '!', " ! " )
        sentence = sentence.replace( '?', " ? " )
        sentence = re.sub( r'http\/\/t\.co\/\w+|\.|\,|\[|\]|&#39;s|\||#|:|;|RT|\(|\)|@\w+|\**', '', sentence )
        sentence = re.sub('de|het|een|van|op|in|http|bij|die|ik|De|tco|dat|over|voor|aan|om', '', sentence)

        sentence = re.sub( ' +',' ', sentence )
        sentence = re.sub(r'''(?ix)\b(?=haha)\S*(\S+)(?<=\bhaha)\1*\b''', 'haha', sentence)
        return sentence

    def __tokenize( self, sentence ):
        return re.findall('\w+|\?|\!', sentence)

    def saveFile(self):
        filename = "Analysis.txt"
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
                doc.write( "Neutral: " + str(sort_neutral[i]) + "\n")
            if( i < len(sort_positive) ):
                doc.write( "Positive: " + str(sort_positive[i]) + "\n")
            if( i < len(sort_negative) ):
                doc.write( "Negative: " + str(sort_negative[i]) + "\n")
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

dataset = csv.reader( open( 'DataCSV.csv', 'rb' ), delimiter=',', quotechar='"' )  
analysis = DataAnalyser( dataset )
analysis.saveFile()
