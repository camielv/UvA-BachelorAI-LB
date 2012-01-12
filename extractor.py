# Extractor extracts a column from a csv file
import csv
import nltk

# Open a file
file = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')

sentence = {}
sentiment = {}

# do not include header!
i = 0

# Collect sentences and sentiments
for entry in file:
    if i == 0:
        i+=1
        continue
    
    # The actual message is the 9th attribute, sentiment is the 4th
    sentence[i - 1] = entry[9]
    sentiment[i - 1] = float(entry[4])
    
    # Stop at 20 (later remove this)
    i += 1
    if ( i == 9999 ): break

number_of_items = i - 1; # -1 because of header  (== len(sentence))

# Create corpus and count word frequencies
corpus = {}

for i in range(number_of_items):
    # Tokenize the sentence
    tk_sentence = nltk.tokenize.word_tokenize( sentence[i] )

    # Iterate over every token
    for token in tk_sentence:
        if token in corpus:
            # Check for sentiment
            if sentiment[i] != 0:
                corpus[token] = corpus[token][0] + 1, corpus[token][1] + 1
            else:
                corpus[token] = corpus[token][0] + 1, corpus[token][1]
        else:
            # Check for sentiment
            if sentiment[i] != 0:
                corpus[token] = 1, 1
            else:
                corpus[token] = 1, 0

# Corpus created, calculate words probability of sentiment based on frequency
probWord = {}
for token in corpus.keys():
    probWord[token] = float(corpus[token][1]) / corpus[token][0]
    # print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', probWord[token], '\n'

# Probability of sentiment per word calculated, estimate sentence probability of sentiment
probSent = {}
confusion = {}
confusion["tp"] = 0
confusion["tn"] = 0
confusion["fp"] = 0
confusion["fn"] = 0

for i in range(len(sentence)):
        p = 1
        tk_sent = nltk.tokenize.word_tokenize( sentence[i] )
        for token in tk_sent:
            p = p + probWord[token]
        probSent[i] = p / float(len(tk_sent)) # to be extra certain intdiv does not occur
        if probSent[i] > 0.31:
            if sentiment[i] == 0:
                confusion["fp"] += 1
            else:
                confusion["tp"] += 1
        if probSent[i] < 0.31:
            if sentiment[i] == 0:
                confusion["tn"] += 1
            else:
                confusion["fn"] += 1
        #print i, 'PROB', probSent[i], 'SENT', sentiment[i]
    
print confusion
print 'accuracy = ', float(confusion["tp"] + confusion["tn"]) / (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
print 'precision = ', float(confusion["tp"]) / (confusion["tp"] + confusion["fp"] )
