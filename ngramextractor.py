# Ngram extractor testfile

str = "a b c d e f g".split(" ")

n = 3;
corpus = {}

# Create temporary dictionary of dictionaries of lists
temp_ngram = {}

for i in range( 1, n + 1 ):
    temp_ngram[i] = {}
    for j in range( 1, i + 1 ):
        temp_ngram[i][j] = []

count = 0;
# Iterate over every word
for word in str:
    count += 1
    # Loop over every n-gram
    for i in range( 1, n + 1 ):
        # Loop over every temporary instantion of an n gram
        for j in range( 1, i + 1 ):
            # Add this word
            if count >= j:
                temp_ngram[i][j].append(word)
            
            if len( temp_ngram[i][j] ) == i:
                # We found a n-gram
                token = tuple(temp_ngram[i][j])
                if token in corpus:
                    corpus[token] += 1
                else:
                    corpus[token] = 1
                # Reset temporary ngram
                temp_ngram[i][j] = []

print corpus
