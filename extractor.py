# Extractor extracts a column from a csv file
import csv
import nltk

# Open a file
file = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')

sentence = {}
sentiment = {}
i = 0

# Collect sentences and sentiments
for entry in file:
    # Do not include the header!
    if i == 0:
        i += 1
        continue
    
    # The actual message is the 9th attribute, sentiment is the 4th
    sentence[i - 1] = entry[9]
    sentiment[i - 1] = float(entry[4])
    
    # Stop at 20 (later remove this)
    i += 1
    if ( i > 200 ): break

number_of_items = i - 1; # -1 because of header

# Create corpus and count word frequencies
corpus = {}

for i in range(number_of_items):
    # Tokenize the sentence
    tk_sentence = nltk.tokenize.word_tokenize( sentence[i] )

    # Iterate over every token
    for token in tk_sentence:
        if token in corpus.keys():
            # Check for sentiment
            if sentiment[i] != 0:
                corpus[token] = corpus[token][0] + 1, corpus[token][1] + 1
            else:
                corpus[token] = corpus[token][0] + 1, 0
        else:
            # Check for sentiment
            if sentiment[i] != 0:
                corpus[token] = 1, 1
            else:
                corpus[token] = 1, 0

# Corpus created, calculate frequencies
frequency = {}
for token in corpus.keys():
    frequency[token] = float(corpus[token][1]) / corpus[token][0]
    print token, ' || ',corpus[token][1],' / ',corpus[token][0],' = ', frequency[token], '\n'

                
