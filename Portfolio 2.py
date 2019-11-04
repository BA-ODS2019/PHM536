# Assignment 1. The collection
# Import TheGuardian OpenApi

import json
import os
from nltk.corpus import stopwords as sw

directory_name = "theguardian/collection/"

ids = list()
texts = list()
sections = list()
for filename in os.listdir(directory_name):
    if filename.endswith(".json"):
        with open(directory_name + filename) as json_file:
            data = json.load(json_file)
            for article in data:
                id = article['id']
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""
                ids.append(id)
                texts.append(text)
                section = article['sectionId']	# Id name each article gets by The Guardian
                sections.append(section) # Adding each item to a list as above "sections = list()"

print("Number of ids: %d" % len(ids))
print("Number of texts: %d" % len(texts))


# Assignment 2. Pre-process and describe your collection
              
sect = set(sections) # Changing the list into a set, meaning that no duplicates of the section titles will appear. It thereby creates a list where each unique name appears only once.
# print(sect) # This could print the whole list of unique categories from the data set.
len(sect) # Counts the list of categories.

# Unique count of each of the ID names with the count of each category. Showing each how many articles there are under the ID name
import numpy as np
unique, counts = np.unique(sections, return_counts=True)
dict(zip(unique, counts))


# How many characters there are combined, through all the data.
all_lengths = list()
for text in texts:
    all_lengths.append(len(text))
print("Total sum of characters in dataset: %i" % sum(all_lengths))

#How many characters there is combined, through all the files.
all_lengths = list()
for text in texts:
    all_lengths.append(len(text))
print("Total sum: %i" % sum(all_lengths))

# When performing a tokenization, we can split up the strings and thereby count the total number of words. This method is without any tokenization tools.
word_count = 0
for text in texts:
    words = text.split()
    word_count = word_count + len(words)
word_count

# To get the unique words, we do a sentence splitter but now also extend the words.
all_words = list()
for text in texts:
  words = text.split()
  all_words.extend(words)
unique_words = set(all_words)
unique_word_count = len(unique_words)
print("Unique word count: %i" % unique_word_count)

#The average word leng is found by first finding all the words individual length, and then calculating the average from that.
total_word_length = 0
for word in all_words:
    total_word_length = total_word_length + len(word)
average_word_length = total_word_length / len(all_words)
print("Average word length: %.6f" % average_word_length)

#To find out how many sentences there are i total, we first make the sentence marker for where a sentence should finish.
def end_of_sentence_marker(character):
    if character in ['.', '?', '!']:#In our case we made '.', '?', '!' our sentence splitters. 
      return True
    else: 
      return False

def split_sentences(texts):
    sentences = []
    start = 0
    for end, character in enumerate(text):#The enumerate turns an iterable in to an object, and can then be used in loops.
        if end_of_sentence_marker(character):
            sentence = text[start: end + 1]
            sentences.append(sentence)
            start = end + 1
    return sentences

all_sentences = list()
for text in texts:
  sentences = split_sentences(text)
  all_sentences.extend(sentences)
sentence_count = len(all_sentences)
sentence_count

#The average number of words in a sentence
words_per_sentence = word_count / sentence_count
print("words_per_sentence: %.6f" % words_per_sentence)

# The following code shows a random sample of the unique words. This is to give an idea of what the words looks like. Here we see that some words are not typical words e.g. numbers ‘£22.99’ or abbreviations like ‘JCRA’ or that the words have punctuation marks around them and therefore appear as a different word than the same word without them: look” vs. look
import random
random.sample(unique_words, 20)


from nltk.tokenize import word_tokenize

# Word tokenization of the documents and converting of big letters to small ones. 
tokens = list()
for text in texts:
  tokens_in_text = word_tokenize(text)
  for token in tokens_in_text:
    if token.isalpha():
      tokens.append(token.lower())


# Creating a new list from the token list with stopwords removed
stopWords = set(sw.words('english'))
swwords = list()
for w in tokens:
    if w not in stopWords:
        swwords.append(w)

print("Str.split word count: %i" % word_count) # Wordcount using str.split
print("Word count: %i" % len(tokens)) # Wordcount using nltk tokenizer
print("Word count with stopwords removed: %i" % len(swwords)) # Wordcount with stopwords removed using nltk tokenizer
unique_tokens = set(tokens)
unique_swwords = set(swwords)
print("Str.split unique word count: %i" % unique_word_count) # Unique wordcount using str.split
print("Unique word count: %i" % len(unique_tokens)) # Unique wordcount using nltk tokenizer
print("Unique word count with stopwords removed: %i" % len(unique_swwords)) # Unique wordcount with stopwords removed using nltk tokenizer

# Here we make a subset of the entire dataset. 2 lists are being made to include both the section id´s: “sport” & “football” and the indexes for the original list. 
idxes = []
subtexts = []
for i, section in enumerate(sections):
    if section in ['sport','football']:
        idxes.append(i)
        subtexts.append(texts[i])
len(idxes) # Test to see how many files there are under the sections “sport” & “football” respectively. These numbers can be compared to the list of all sections, where the total number of articles per section can be observed. 

# Assignment 3. Select articles using a query
# To create the document-term matrix for the dataset we need to use CountVectorizer, 
# which can be imported with sklearn. To transform our data we must also use model_vect.fit_transform.
# When running to code, it tells us that there are 2.395.815 elements in our document matrix 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as sw
model_vect = CountVectorizer(stop_words=sw.words('english'), token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vect = model_vect.fit_transform(subtexts)
print('Shape: (%i, %i)' % data_vect.shape) # This shows how many documents 6808 x how many terms 87615 there are in our subset. We changed the initial dataset from only containing articles from starting on the 1st of September to ending on the 30th of October, to starting on the 1st of April to ending on the 30th of October. We did so because we wanted to show our results on a larger datascale.  
data_vect

# In this line of code we demonstrate how we're able to find the index placement of the 10 most used
# words in our documents. When using .A1 we're able to present results in an array, which
# can be described as a line of numbers. The [:10] tells that we want it as a top 10. We can change this
# number if we want to show more or fewer results eg. 20 or 5. 
counts = data_vect.sum(axis=0).A1
top_idxs = (-counts).argsort()[:10]
top_idxs

# Here we use inverted_vocabulary to assign the actual words belonging to the top-10 indexes in the sub-sets.
inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words in subset: %s" % top_words)

# Frequency/amount of times the words are being used in the entire dataset. So we can compare it to the most used word in the subsets.
from nltk.probability import FreqDist
fdist = FreqDist(swwords)
fdist.most_common(30)

# This line is not important in itself, but it creates a submatrix that will be used later on.
import random
some_row_idxs = random.sample(range(0,len(subtexts)), 10)
print("Selection: (%s x %s)" % (some_row_idxs, top_idxs))
sub_matrix = data_vect[some_row_idxs, :][:, top_idxs].todense()
sub_matrix

# The code here transforms the tf-idf model.
from sklearn.feature_extraction.text import TfidfTransformer
model_tfidf = TfidfTransformer()
data_tfidf = model_tfidf.fit_transform(data_vect)

# The freqs function helps us sort the top 10 words.
freqs = data_tfidf.mean(axis=0).A1
top_idxs = (-freqs).argsort()[:10].tolist()
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
(top_idxs, top_words)

# Now we can use the submatrix and the transformed tf-idf, to create a scheme of the top 10 most used words and their weight in 10 random documents.
import pandas as pd
sub_matrix = data_tfidf[some_row_idxs, :][:,top_idxs].todense()
df = pd.DataFrame(columns=top_words, index=some_row_idxs, data=sub_matrix)
df

# Query terms that we think we can find in our subsets. 
terms = ['tottenham', 'hotspurs', 'hotspur', 'spurs', 'stadium', 'new', 'home']
terms

# The count of each term in the subset, how many times it is used. 
term_idxs = [model_vect.vocabulary_.get(term) for term in terms]
term_counts = [counts[idx] for idx in term_idxs]
term_counts

# Calculate the term weights 
idfs = model_tfidf.idf_
term_idfs = [idfs[idx] for idx in term_idxs]
term_idfs

# Creates a matrix for the query words and their weight in the choosen subsets
dfi = pd.DataFrame(columns=['count', 'idf'], index=terms, data=zip(term_counts,term_idfs))
dfi

# We will now try to answer our research question: With which topics is the football club Tottenham mentioned? from previously we have the terms used in assignment 3. First we combine the 7 words.
query = " ".join(terms)
query

# Then changing the query in to a sparse matrix with 1 row. 
query_vect_counts = model_vect.transform([query])
query_vect = model_tfidf.transform(query_vect_counts)
query_vect

# We then compare all terms with our research question, where the percentage on how well they fit our query.
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(query_vect, data_tfidf)
sims

# Here we sort the documents on how well they fit our query, where the most likely is in the top.
sims_sorted_idx = (-sims).argsort()
sims_sorted_idx

# The document that fits our query the most will now be shown and you get the whole article. the article is https://www.theguardian.com/football/blog/2019/apr/03/tottenham-new-stadium-spurs-numbers-game-crystal-palace which is about the new stadium that Tottenham hotspur’s just got in March. 
subtexts[sims_sorted_idx[0,0]]

# Mini matrix showing cosine similarity between articles and research question showing which article is most likely to have the words from our question. 
print("Mini matrix showing cosine similarity: (%i, %i)" % (len(sims), len(sims[0,:])) )
df = pd.DataFrame(data=zip(sims_sorted_idx[0,:], sims[0,sims_sorted_idx[0,:]]), columns=["Section Id", "Cosine Similarity"])
df[0:10]

