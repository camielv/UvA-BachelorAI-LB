import sys;
args = sys.argv
# TODO: Moet nog aangepast worden voor de juiste datasets
# INPUT PARSING
if (len(args) < 3 or (len(args) > 1 and int(args[1]) <= 0 ) or (len(args) > 2 and args[2] == "")):
    print "BAD REQUEST"
# RUN CODE
else:
    dataset = args[1]
    message = args[2]
    # Dataset ID
    print dataset
    # Message
    print message
    # Sentiment
    print 1
    # Accuracy
    print 80
    # Precision
    print 96
    # Recall
    print 90
