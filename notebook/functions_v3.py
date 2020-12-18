import sys
import pandas as pd
import json
from array import array
from collections import defaultdict
import nltk
import numpy as np
from numpy import linalg as la



nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet = True)


from nltk.corpus import stopwords
from collections import Counter
from config import *
from nltk.stem import PorterStemmer
import re
import time
import math
import collections


def getTerms(line):
        
    stemming = PorterStemmer()
    stops = set(stopwords.words("english"))
    
    line = line.replace("RT", "").strip()# remove "RT" string indicating a retweet
    line = line.replace("#", "").strip()# remove "#" string
    line=  line.lower() ## Transform in lowercase
    line= re.sub(r'[^\w\s]','',line).strip()    # removing all the punctuations
    line= line.encode('ascii', 'ignore').decode('ascii')#Remove emojis

    line=  line.split() ## Tokenize the text to get a list of terms
    line=[x for x in line if x not in stops]  ##eliminate the stopwords 
    #line=[stemming.stem(word) for word in line] 
    #We have decided to lemmatize instead of stemming since we have tried both methods and this one works better in this case.
    wnl = nltk.WordNetLemmatizer()
    line=[wnl.lemmatize(word) for word in line]
    
    return line

def get_bagofwords(data, attribute):
  bag_of_words = {}
  STOPWORDS = set(stopwords.words("english"))
  for tweet in data[attribute]:
    for text in tweet:
      # remove "RT" string indicating a retweet
      text = text.replace("RT", "").strip()
      text = text.replace("#", "").strip()
      # lowering text
      text = text.lower()
      
      # removing all the punctuations
      text = re.sub(r'[^\w\s]','',text).strip()
      text= text.encode('ascii', 'ignore').decode('ascii')
      
      # tokenize the text
      lst_text = text.split()
      
      # remove stopwords
      lst_text = [x for x in lst_text if x not in STOPWORDS]
          
      # create bag-of-words - for each word the frequency of the word in the corpus
      for w in lst_text:
          if w not in bag_of_words:
              bag_of_words[w] = 0
          bag_of_words[w]+=1
  return bag_of_words


  #Function to plot the wordcloud
def plot_wordcloud(title, dic_):
    fig, ax = plt.subplots(1, 1, figsize=(18,7))
    wordcloud = WordCloud(background_color="white",width=1600, height=800)
    wordcloud = wordcloud.generate_from_frequencies(dic_)
    ax.axis("off")     
    ax.imshow(wordcloud, interpolation='bilinear')

    ax.set_title(title)
    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()


#Create index
def create_tf_idf_index(data, col, numDocuments):
    """
    Implement the inverted index and compute tf, df and idf
    
    Argument:
    data: dataframe with all the tweets
    col: the column where the tweet text is located
    numDocuments -- total number of tweets
    
    Returns:
    index - the inverted index (implemented through a python dictionary) containing terms as keys and the corresponding 
    list of tweets these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each tweet
    df - number of tweets each term appear in
    idf - inverse document frequency of each term
    """
       
    index=defaultdict(list)
    tf=defaultdict(list) #term frequencies of terms in tweets (tweets in the same order as in the main index)
    df=defaultdict(int)         #document frequencies of terms in the corpus
    idf=defaultdict(float)
    ## ===============================================================        
        ## create the index for the **currenttweet** and store it in termdict
        ## termdict ==> { ‘term1’: [currenttweet, [list of positions]], ...,‘termn’: [currenttweet, [list of positions]]}
        
        ## Example: if the currenttweet has id 1 and his text is "web retrieval information retrieval":
        ## termdict==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}
        ## the term ‘web’ appears in tweet 1 in positions 0, 
        ## the term ‘retrieval’ appears in tweet 1 in positions 1 and 4
        ## ===============================================================
    for i, row in data.iterrows():#iterate the dataframe
        row_id = i

        termdict={} #here we store a dictionary from the word to a list with the current tweet and the postings
        for position, term in enumerate(row[col]): 
          try:# if the term is already in the dict append the position to the corresponding list
            termdict[term][1].append(position)
          except:# Add the new term as dict key and initialize the array of positions and add the position
            termdict[term]=[row_id, ([position])] 
        
        # Compute the denominator to normalize term frequencies
        # norm is the same for all terms of a document.
        norm=0
        for term, postings in termdict.items():
            # postings is a list containing tweet_id and the list of positions for current term in current tweet: 
            # posting ==> [tweet_id, [list of positions]] 
            # you can use it to inferr the frequency of current term  
            norm+=len(postings[1])**2

        norm=math.sqrt(norm)
        

        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in termdict.items():     
            # append the tf for current term (tf = term frequency in current tweet/norm)
            tf[term].append(np.round(len(posting[1])/norm,4))  
            #increment the tweet frequency of current term (number of tweets containing the current term)
            df[term]+=1  # increment df for current term          
        
        #merge the current page index with the main index
        for term, posting in termdict.items():
          index[term].append(posting)
            
        # Compute idf 
    for term in df:
        idf[term] = np.round(np.log(float(numDocuments/df[term])),4)

    return index, tf, df, idf

#Define functions to rank documents 
def rankDocuments(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    terms -- list of query terms
    docs -- list of tweets, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted tweets frequencies
    tf -- term frequencies
    
    Returns:
    Print the list of ranked tweets
    """
        
    # We're interested only on the element of the docVector corresponding to the query terms 
    # The remaing elements would became 0 when multiplied to the queryVector
    docVectors=defaultdict(lambda: [0]*len(terms)) # We call docVectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    queryVector=[0]*len(terms)    

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms) # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    
    query_norm = la.norm(list(query_terms_count.values()))
    
    for termIndex, term in enumerate(terms): #termIndex is the index of the term in the query
        if term not in index:
            continue
                    
        ## Compute tf*idf(normalize tf as done with documents)
        queryVector[termIndex]=query_terms_count[term]/query_norm * idf[term] 

        # Generate docVectors for matching tweets
        for docIndex, (doc, postings) in enumerate(index[term]):
            # Example of [docIndex, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # term is in tweet 26 in positions 1,4, .....
            
            #tf[term][0] will contain the tf of the term "term" in the tweet 26            
            if doc in docs:
                docVectors[doc][termIndex]=tf[term][docIndex] * idf[term] 

    # calculate the score of each tweet
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine siilarity
    # see np.dot
    
    docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]
    docScores.sort(reverse=True)
    resultDocs=[x[1] for x in docScores]
    
    while len(resultDocs) == 0:
        print("No results found, try again")
        query = input()
        resultDocs = search_tf_idf(query, index, tf, idf)    

    return resultDocs

def search_tf_idf(query, index, tf, idf):
    '''
    output is the list of tweets that contain all of the query terms. 
    So, we will get the list of tweets for each query term, and take the AND of them.
    '''
    query=getTerms(query)
    docs= set()
    for term in query:
        try:
            # store in termDocs the ids of the docs that contain "term"                        
            termDocs=[posting[0] for posting in index[term]]

            if len(docs)==0:
              docs = set(termDocs)
            else:
              docs= docs & set(termDocs)
            
            # docs = docs Union termDocs
           
        except:
            #term is not in index
            pass
    docs=list(docs)
    ranked_docs = rankDocuments(query, docs, index, idf, tf)   
    return ranked_docs

def search_ourRanking(query, index, idf, tf):
    '''
    The output is the list of tweets that contain the query terms. 
    So, we will get the list of tweets for each query term, and take the union of them.
    '''
    query=getTerms(query)

    docs= set()
    for term in query:
        try:
            # store in termDocs the ids of the tweets that contain "term"                        
            termDocs=[posting[0] for posting in index[term]]

            if len(docs)==0:
              docs = set(termDocs)
            else:
              docs= docs & set(termDocs)
            
           
        except:
            #term is not in index
            pass

    docs=list(docs)
    ranked_docs = ourRankDocuments(query, docs, index, idf, tf)   
    
    return ranked_docs

def ourRankDocuments(terms, docs, index, idf, tf):
  
  """
    Perform the ranking of the results of a search based on the tf-idf weights and popularity of the tweet (number of likes and retweets)
    
    Argument:
    terms -- list of query terms
    docs -- list of tweets, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted tweets frequencies
    tf -- term frequencies
    
    Returns:
    Print the list of ranked tweets
    """  

  """It is the same code that for the normal ranking function"""
    # I'm interested only on the element of the docVector corresponding to the query terms 
    # The remaing elements would became 0 when multiplied to the queryVector
  docVectors=defaultdict(lambda: [0]*len(terms)) 
    # I call docVectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
  queryVector=[0]*len(terms)    
  query_terms_count = collections.Counter(terms) # get the frequency of each term in the query. 
    # compute the norm for the query tf
  query_norm = la.norm(list(query_terms_count.values()))
    
  for termIndex, term in enumerate(terms): #termIndex is the index of the term in the query
      if term not in index:
          continue
                    
      ## Compute tf*idf(normalize tf as done with documents)
      queryVector[termIndex]=query_terms_count[term]/query_norm * idf[term] 
       
        # Generate docVectors for matching docs
      for docIndex, (doc, postings) in enumerate(index[term]):
            # Example of [docIndex, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....  # term is in doc 33 in positions 26,33, .....
            #tf[term][0] will contain the tf of the term "term" in the doc 26           

          if doc in docs: #if the docment is in the intersection
            #docVectors[doc][termIndex]=tf[term][docIndex] * 0.25 + df_proc.iloc[docIndex]["Likes"]*0.25 + df_proc.iloc[docIndex]["Retweets"]*0.25 +  idf[term]*0.25 # TODO: check if multiply for idf
            #docVectors[doc][termIndex]=tf[term][docIndex] * popularity[term]
            '''The main change is here, since instead of giving the score to a tweet based only on the tf-idf score, we 
            also take into account the number of likes and retweets it had, showing first the popular tweets.'''
            docVectors[doc][termIndex]=tf[term][docIndex] * idf[term]*0.6 + df_proc.iloc[doc]["Likes"]*0.15 + df_proc.iloc[doc]["Retweets"]*0.25
    
    # calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
  docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]        
  docScores.sort(reverse=True)

  resultDocs=[x[1] for x in docScores]
    
  while len(resultDocs) == 0:
      print("No results found, try again")
      query = input()
      resultDocs = search_ourRanking(query, index, idf, tf)   
    #print ('\n'.join(resultDocs), '\n')
  return resultDocs