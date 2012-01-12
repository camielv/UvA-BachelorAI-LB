# import nltk
import nltk.tokenize

# Corpus
corpus = 'blaat bla blaa bla bal bocht hendrik kees'
bagOfWords = nltk.tokenize.word_tokenize( corpus )

# List of vectors
vecs = []

# List of sentences
sentences = []
sentences.append('kees blaat hendrik')
sentences.append('bla blaa bal')
sentences.append('hendrik bocht bal')

for sentence in sentences:
    vec = []

    # Tokenize sentence
    tk_sentence = nltk.tokenize.word_tokenize( sentence )

    # Create vector
    for word in bagOfWords:
        if word in tk_sentence:
            vec.append(1)
        else:
            vec.append(0)
    vecs.append(vec)

# Print matrix
for vec in vecs:
    print vec