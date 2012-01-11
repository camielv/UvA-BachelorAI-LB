#import nltk
import nltk.tokenize

corpus = 'blaat bla blaa bla bal bocht hendrik kees'
bagOfWords = nltk.tokenize.word_tokenize( corpus )

vec = []
sentence = 'kees blaat hendrik'
tk_sentence = nltk.tokenize.word_tokenize( sentence )
for word in bagOfWords:
    if word in tk_sentence:
        vec.append(1)
    else:
        vec.append(0)

print vec
        
