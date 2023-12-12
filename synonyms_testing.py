from nltk.corpus import wordnet

synonyms = []

for syn in wordnet.synsets('evaporated milk'):
    for i in syn.lemmas():
        synonyms.append(i.name())

print(set(synonyms))


from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
 
print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))
 
# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos ="a"))

print("chicken breasts :", lemmatizer.lemmatize("chicken breasts", pos ="n"))

test = "chicken breasts"
test = test.split()
for word in test:
    w = lemmatizer.lemmatize(word, pos='n')
    print(w)

