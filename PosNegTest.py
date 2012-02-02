import cPickle as pickle
import re
import inputDevice as iD

# Number of classes and distribution
class_distribution = [0],[-1,-2],[1,2]
display_class = 1,2
num_classes = len( class_distribution )

print 'Loading corpus'
corpus = pickle.load( open('FeatureProbs.txt') )

n = 4

def tokenize( sentence ):  
    return re.findall( '\w+|\?|\!', sentence )

def clean( sentence ):
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

while 1:
    test_sentence = raw_input('Typ een zin: ')
    # Calculate probabilities for every sentence

    tk_sent = tokenize( clean( test_sentence ) )
    # Create temporary dictionary of dictionaries of lists
    temp_ngram = {}

    for k in range( 1, n + 1 ):
        temp_ngram[k] = {}
        for j in range( 1, k + 1 ):
            temp_ngram[k][j] = []

    # Counter used for ngram finding
    count = 0

    # Number of found features
    features = 0

    # Running probability totals
    classes = [ 0 for c in range( num_classes ) ]

    # Iterate over every word
    for word in tk_sent:
        count += 1
        # Loop over every n-gram
        for k in range( 1, n + 1 ):
            # Loop over every temporary instantion of an n gram
            for j in range( 1, k + 1 ):
                # Add this word
                if count >= j:
                    temp_ngram[k][j].append( word )
                
                if len( temp_ngram[k][j] ) == k:
                    # We found a n-gram
                    token = tuple( temp_ngram[k][j] )

                    # Check if the token is in the corpus
                    if token in corpus:
                        for c in range( num_classes ):
                            # If so, add the chances to the probabilities
                            classes[c] += float( corpus[token][c] ) / sum( corpus[token] )
                            features += 1
                            
                    temp_ngram[k][j] = []
                # Normalize to the number of features
    try:
        for c in range( num_classes ):
            classes[c] /= float( features )
    except:
        print 'zero features found in following sentence:'
        print test_sentence
    
    print test_sentence

    opinion = iD.classifyNewLine(test_sentence, './weights500s5000i.txt')
    print 'Opinion probability: {0}'.format(opinion)
    print 'Negative Probability: {0}\nPositive Probability: {1}'.format(classes[1],classes[2])

    if opinion > 0.5:
        if classes[1] > classes[2]:
            print 'Sentence is negative'
        elif classes[1] < classes[2]:
            print 'Sentence is positive'
        else:
            print 'Equal probability, unsure' 
    else:
        if classes[1] > classes[2]:
            posneg = 'negative'
        elif classes[1] < classes[2]:
            posneg = 'positive'
        else:
            posneg = 'unsure whether it is positive or negative.'
        print 'Sentence classified as neutral, but if it is not, it would be ' + posneg + '.'
