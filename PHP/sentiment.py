# Filename: sentiment.py
# Written by: Camiel Verschoor
# Definition: Runs the classifier on the user input

import sys
import CF

# Read out command line arguments
args = sys.argv

# Parse input

# If wrong return a bad request
if (len(args) < 3 or (len(args) > 1 and int(args[1]) <= 0 ) or (len(args) > 2 and args[2] == "")):
    print "BAD REQUEST"

# Else return the sentiment value
else:
    dataset = args[1]
    message = args[2]

    # Dataset ID
    print dataset
    
    print message
        
    Classifier = CF.Classifier()
    (success, sentiment, P_sentiment, certainty_sentiment) = Classifier.classify( message )

    if( sentiment == 0 or success ):
        print "True"
        print sentiment
        print certainty_sentiment
    else:
        print "False"
        for i in range(2):
            print 0
