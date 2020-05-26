#import libraries
import pandas as pd
pd.set_option('display.max_colwidth', 100)

#import the data file
data = pd.read_csv("C:/Users/nisht/Anaconda3_n/envs/LL/NLP/Ch01/01_10/End/SMSSpamCollection.tsv", sep='\t', header=None)
data.columns = ['label', 'body_text']

#remove punctuations
import string
string.punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))
data.head()

#Tokenization
import re
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))
data.head()

#Remove stopwords
import nltk
stopword= nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
data.head()

#Stemming
ps=nltk.PorterStemmer()
def stemming(tokenized_text):
    text= [ps.stem(word) for word in tokenized_text]
    return text
data['body_text_stemmed']= data['body_text_nostop'].apply(lambda x: stemming(x))
data.head()
