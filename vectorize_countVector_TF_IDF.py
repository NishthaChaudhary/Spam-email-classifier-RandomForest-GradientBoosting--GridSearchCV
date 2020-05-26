# Read Text
import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth', 100)
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
data = pd.read_csv("C:/Users/nisht/Anaconda3_n/envs/LL/NLP/Ch01/01_10/End/SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

# Create function to remove punctuation, tokenize, remove stopwords, and stem
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
    
# Apply CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())

X_counts = pd.DataFrame(X_counts.toarray())
X_counts

X_counts.columns = count_vect.get_feature_names()
X_counts

# Apply TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())

X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
X_tfidf_df.columns = tfidf_vect.get_feature_names()
X_tfidf_df

